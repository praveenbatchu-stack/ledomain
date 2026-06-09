"""
LE pipeline — canonical domain ↔ LE mapping + accuracy.

Extracted from app.py so it can be reused outside Streamlit (batch runs,
tests, automation). app.py imports the same names from here, so editing
the pipeline updates the console too.

Functions:
  _domain_alive(domain)                          → bool
  playwright_fetch_domain_pages(domain)          → str
  _normalize_le_name(name)                       → str
  _find_le_via_search(domain, search_type, q, country) → dict
  _find_le_via_webfetch(domain, country)         → dict
  find_le_from_domain(domain, country)           → dict
  _webfetch_with_text(entry, page_text)          → dict
  run_accuracy_check_single(domain, le_name, country, cin) → dict
"""
import os, re, time, threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import socket
import requests

from domain import (
    check_forward, check_reverse, check_webfetch,
    compute_final_verdict, ai_call, parse_json_safe,
    web_search, fetch_domain_pages,
    names_are_equivalent, fuzzy_name_match,
    GOOD_VERDICTS as _GOOD_VERDICTS,
    _resolve_mismatch_via_search,
)


# ---------------------------------------------------------------------------
# Liveness pre-check — thread-safe (no socket.setdefaulttimeout)
# ---------------------------------------------------------------------------
def _domain_alive(domain: str, timeout: int = 4) -> bool:
    host = domain.split('/')[0]
    # DNS — getaddrinfo with one retry on EAI_AGAIN
    for attempt in range(2):
        try:
            socket.getaddrinfo(host, None)
            break
        except socket.gaierror as e:
            if attempt == 0 and getattr(e, 'errno', 0) in (-3, 11):
                time.sleep(0.4)
                continue
            return False
        except Exception:
            return False
    for scheme in ('https', 'http'):
        try:
            r = requests.head(f'{scheme}://{domain}', timeout=timeout,
                              allow_redirects=True,
                              headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code < 500:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Playwright fetch (headless) with anchor-link discovery
# ---------------------------------------------------------------------------
def playwright_fetch_domain_pages(domain: str) -> str:
    if not _domain_alive(domain):
        return ''
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ''

    base = f'https://{domain}'
    collected = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            discovered_links = []
            try:
                page.goto(base, timeout=10000, wait_until='domcontentloaded')
                page.wait_for_timeout(1500)
                links = page.eval_on_selector_all('a[href]', """
                    els => els.map(el => ({href: el.href, text: el.textContent.trim().toLowerCase()}))
                """)
                for link in links:
                    href, text = link.get('href', ''), link.get('text', '')
                    if any(kw in text for kw in ['privacy', 'legal', 'terms',
                                                 'about us', 'about']):
                        if href and href.startswith('http'):
                            discovered_links.append(href)
                    elif any(kw in href.lower() for kw in ['privacy', 'legal',
                                                           'terms', 'about']):
                        if href and href.startswith('http'):
                            discovered_links.append(href)
                text = page.inner_text('body')
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    collected.append(f'=== / (playwright) ===\n{text[:4000]}')
            except Exception:
                pass

            all_urls = list(dict.fromkeys(discovered_links[:3]))
            for path in ['/privacy-policy', '/privacy', '/about']:
                url = urljoin(base, path)
                if url not in all_urls and url != base:
                    all_urls.append(url)

            for url in all_urls:
                if len(collected) >= 2:
                    break
                try:
                    page.goto(url, timeout=8000, wait_until='domcontentloaded')
                    page.wait_for_timeout(1000)
                    text = page.inner_text('body')
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text and len(text) > 50:
                        label = urlparse(url).path or url
                        collected.append(f'=== {label} (playwright) ===\n{text[:4000]}')
                except Exception:
                    pass
            browser.close()
    except Exception:
        pass
    return '\n\n'.join(collected)[:12000]


# ---------------------------------------------------------------------------
# LE NAME NORMALIZATION + 3-STEP LE DISCOVERY
# ---------------------------------------------------------------------------
def _normalize_le_name(name):
    n = (name or '').lower().strip()
    n = re.sub(r'\bltd\.?\b',  'limited', n)
    n = re.sub(r'\bp\.?l\.?c\.?\b', 'plc', n)
    n = re.sub(r'\bl\.?l\.?p\.?\b', 'llp', n)
    n = re.sub(r'\binc\.?\b',  'incorporated', n)
    n = re.sub(r'\bcorp\.?\b', 'corporation', n)
    n = re.sub(r'[^a-z0-9 ]', '', n)
    return re.sub(r'\s+', ' ', n).strip()


def _find_le_via_search(domain, search_type, query, country=''):
    results = web_search(query)
    if not results:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': f'{search_type}: no results', 'country': country}
    ctx = '\n'.join(
        f"[{i+1}] URL: {r['url']}\n    Title: {r.get('title','')}\n    Snippet: {r.get('snippet','')}"
        for i, r in enumerate(results[:8]))
    country_hint = f' in {country}' if country else ''
    prompt = f"""Find the Legal Entity (LE) that OWNS and OPERATES the domain "{domain}"{country_hint}.

Search results ({search_type}):
{ctx}

TASK:
1. Identify the REGISTERED LEGAL NAME of the company that owns "{domain}"
2. Find its company registration number if available

RULES:
- Return the EXACT registered legal name (e.g. "WISE PAYMENTS LIMITED" not "Wise")
- Include legal suffix (Ltd, Limited, PLC, LLP, Inc, Pvt Ltd, etc.)
- If unsure, return empty — false positive is worse than a miss
- Include the country of registration if known

Respond ONLY with JSON:
{{"le_name": "EXACT REGISTERED LEGAL NAME", "cin": "company number or empty",
  "country": "country of registration or empty",
  "confidence": "high|medium|low", "reason": "brief"}}"""
    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=400))
        return {
            'le_name':    (r.get('le_name') or '').strip(),
            'cin':        str(r.get('cin') or '').strip(),
            'country':    (r.get('country') or country).strip(),
            'confidence': r.get('confidence', 'low'),
            'reason':     r.get('reason', ''),
        }
    except Exception as e:
        return {'le_name': '', 'cin': '', 'confidence': 'error',
                'reason': str(e), 'country': country}


def _find_le_via_webfetch(domain, country=''):
    page_text = playwright_fetch_domain_pages(domain)
    if not page_text or len(page_text.strip()) < 50:
        page_text = fetch_domain_pages(domain)
    if not page_text or len(page_text.strip()) < 50:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': 'could not fetch website', 'country': country}
    country_hint = f' in {country}' if country else ''
    prompt = f"""You fetched pages from "{domain}". Find the Legal Entity that owns this domain{country_hint}.

WEBSITE CONTENT (excerpts from /, /about, /privacy-policy):
{page_text[:7000]}

TASK:
1. Look in footer, privacy policy, terms, about page for:
   - Registered company name (legal name with Ltd/Limited/PLC/Inc/Pvt Ltd suffix)
   - Company registration number
   - Country of registration
2. Return the entity that directly owns this domain

RULES:
- Return EXACT registered legal name with suffix
- If multiple entities shown, pick the one that directly owns the domain
- If no entity found, return empty
- Include country of registration

Respond ONLY with JSON:
{{"le_name": "EXACT REGISTERED LEGAL NAME", "cin": "company number or empty",
  "country": "country of registration or empty",
  "confidence": "high|medium|low", "reason": "brief"}}"""
    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=400))
        return {
            'le_name':    (r.get('le_name') or '').strip(),
            'cin':        str(r.get('cin') or '').strip(),
            'country':    (r.get('country') or country).strip(),
            'confidence': r.get('confidence', 'low'),
            'reason':     r.get('reason', ''),
        }
    except Exception as e:
        return {'le_name': '', 'cin': '', 'confidence': 'error',
                'reason': str(e), 'country': country}


def find_le_from_domain(domain, country=''):
    """3-step LE discovery: forward search + reverse search + webfetch.
    Agreement boosts confidence."""
    country_hint = f' {country}' if country else ''
    fwd = _find_le_via_search(domain, 'forward',
        f'"{domain}"{country_hint} company registration legal entity official', country)
    rev = _find_le_via_search(domain, 'reverse',
        f'site:{domain} "registered in" OR "company number" OR "registered office"', country)
    web = _find_le_via_webfetch(domain, country)

    candidates = [fwd, rev, web]
    found = [c for c in candidates if c.get('le_name')]
    if not found:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': 'No LE found from any method (forward/reverse/webfetch)',
                'country': country,
                'fwd_le': '', 'rev_le': '', 'web_le': ''}

    reason_parts = []
    if web.get('le_name'):
        best = web
        le_name = web['le_name']
        cin = web.get('cin', '')
        found_country = web.get('country', country)
        reason_parts.append(f"webfetch: {web.get('reason', '')}")
        agree_count = 1
        for other in [fwd, rev]:
            if other.get('le_name') and _normalize_le_name(other['le_name']) == _normalize_le_name(le_name):
                agree_count += 1
        if agree_count >= 2:
            confidence = 'high'
            reason_parts.append(f'{agree_count}/3 methods agree')
        else:
            confidence = web.get('confidence', 'medium')
            reason_parts.append('webfetch authority')
        for label, other in [('forward', fwd), ('reverse', rev)]:
            if other.get('le_name') and _normalize_le_name(other['le_name']) != _normalize_le_name(le_name):
                reason_parts.append(f'{label} disagrees: {other["le_name"]}')
    else:
        if fwd.get('le_name') and rev.get('le_name') and \
           _normalize_le_name(fwd['le_name']) == _normalize_le_name(rev['le_name']):
            best = fwd if fwd.get('cin') else rev
            le_name = best['le_name']
            cin = best.get('cin', '')
            found_country = best.get('country', country)
            confidence = 'medium'
            reason_parts.append('fwd+rev agree (no webfetch)')
        else:
            candidates_sorted = sorted(found,
                key=lambda c: ('high', 'medium', 'low', 'none').index(c.get('confidence', 'none')))
            best = candidates_sorted[0]
            le_name = best['le_name']
            cin = best.get('cin', '')
            found_country = best.get('country', country)
            confidence = 'low'
            reason_parts.append(f"best single: {best.get('reason', '')}")
            for c in found:
                if _normalize_le_name(c['le_name']) != _normalize_le_name(le_name):
                    reason_parts.append(f'DISAGREEMENT: also found: {c["le_name"]}')

    return {
        'le_name': le_name, 'cin': cin, 'country': found_country,
        'confidence': confidence, 'reason': ' | '.join(reason_parts),
        'fwd_le': fwd.get('le_name', ''),
        'rev_le': rev.get('le_name', ''),
        'web_le': web.get('le_name', ''),
    }


# ---------------------------------------------------------------------------
# 3-CHECK ACCURACY (runs after we have a candidate LE name)
# ---------------------------------------------------------------------------
def _webfetch_with_text(entry, page_text):
    kd, kle = entry['known_domain'], entry['known_le_name']
    eid, cty = entry.get('entity_id', ''), entry.get('country', '')
    if not page_text.strip():
        return {'webfetch_legal_name': '', 'webfetch_company_num': '',
                'webfetch_parent': '', 'webfetch_verdict': 'FETCH_FAILED',
                'webfetch_confidence': '', 'webfetch_explanation': ''}
    prompt = f"""You fetched pages from "{kd}". Verify if this domain belongs to the legal entity below.

EXPECTED ENTITY:
  Name:            {kle}
  Registration ID: {eid}
  Country:         {cty}

WEBSITE CONTENT:
{page_text[:7000]}

VERDICT — choose exactly ONE:
  "EXACT_MATCH"         : site confirms this is exactly "{kle}"
  "NAME_CHANGED"        : same entity but renamed
  "PARENT_MATCH"        : domain belongs to parent/holding of "{kle}"
  "BRAND_NAME_MISMATCH" : domain correct but "{kle}" is brand name, not registered name
  "POSSIBLE_MISMATCH"   : site shows a DIFFERENT name — relationship unclear
  "MISMATCH"            : domain clearly belongs to an UNRELATED company
  "NOT_FOUND"           : not enough info on site to determine

Respond ONLY with JSON:
{{
  "legal_name_on_site": "name from website or empty",
  "company_num_on_site": "company number or empty",
  "parent_company": "parent company name if mentioned, else empty",
  "verdict": "EXACT_MATCH|NAME_CHANGED|PARENT_MATCH|BRAND_NAME_MISMATCH|POSSIBLE_MISMATCH|MISMATCH|NOT_FOUND",
  "confidence": "high|medium|low",
  "explanation": "1-2 sentence plain-English explanation of verdict"
}}"""
    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=800))
        result = {
            'webfetch_legal_name':  r.get('legal_name_on_site', ''),
            'webfetch_company_num': r.get('company_num_on_site', ''),
            'webfetch_parent':      r.get('parent_company', ''),
            'webfetch_verdict':     r.get('verdict', 'NOT_FOUND'),
            'webfetch_confidence':  r.get('confidence', 'low'),
            'webfetch_explanation': r.get('explanation', ''),
        }
        if result['webfetch_verdict'] in ('POSSIBLE_MISMATCH', 'MISMATCH'):
            site_name = result.get('webfetch_legal_name', '')
            if site_name and names_are_equivalent(site_name, kle):
                result['webfetch_verdict'] = 'EXACT_MATCH'
                result['webfetch_explanation'] = (
                    f"Names match after normalization: '{site_name}' ≡ '{kle}'")
            elif site_name and fuzzy_name_match(site_name, kle) in _GOOD_VERDICTS:
                fm = fuzzy_name_match(site_name, kle)
                result['webfetch_verdict'] = fm
                result['webfetch_explanation'] = (
                    f"Names are a {fm.lower().replace('_',' ')}: "
                    f"'{site_name}' vs '{kle}'")
            else:
                result = _resolve_mismatch_via_search(entry, result)
        return result
    except Exception as e:
        return {'webfetch_legal_name': '', 'webfetch_company_num': '',
                'webfetch_parent': '', 'webfetch_verdict': 'ERROR',
                'webfetch_confidence': '', 'webfetch_explanation': str(e)}


def run_accuracy_check_single(domain, le_name, country='', cin=''):
    """3-check accuracy: forward + reverse + webfetch (Playwright primary)."""
    entry = {'entity_id': cin or '', 'country': country or '',
             'known_domain': domain, 'known_le_name': le_name}
    try:
        with ThreadPoolExecutor(max_workers=4) as inner:
            f_fwd = inner.submit(check_forward, entry)
            f_rev = inner.submit(check_reverse, entry)
            f_pw  = inner.submit(playwright_fetch_domain_pages, domain)
            fwd, rev = f_fwd.result(), f_rev.result()
            pw_text = f_pw.result()
        if pw_text and pw_text.strip():
            web = _webfetch_with_text(entry, pw_text)
        else:
            web = check_webfetch(entry)
        merged = {**entry, **fwd, **rev, **web}
        final = compute_final_verdict(merged)
        return {**merged, **final}
    except Exception as e:
        return {'final_mapping_correct': 'NEEDS_REVIEW',
                'final_issue_notes': str(e)}
