"""
Domain LE Console — Streamlit App
==================================
Two modes:
  1. Find LE (Domain Mapping): Given domains → find legal entity + optionally verify on CH (UK only)
  2. Check Accuracy: Given domain + LE pairs → run 3-check accuracy pipeline

Input: Upload CSV/Excel OR connect Google Sheet
Output: Results table + download + optional write-back to Google Sheet

Deploy: Streamlit Community Cloud — secrets in .streamlit/secrets.toml
"""

import os
import re
import time
import threading
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# SECRETS → env vars (must happen before importing domain.py)
# Secrets are loaded from Streamlit secrets (.streamlit/secrets.toml)
# or environment variables. Never hardcode credentials here.
# ---------------------------------------------------------------------------
import json as _json

if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = st.secrets.get("NVIDIA_API_KEY", "")

from domain import (
    check_forward, check_reverse, check_webfetch,
    compute_final_verdict, ai_call, parse_json_safe,
    web_search, fetch_domain_pages,
    names_are_equivalent, fuzzy_name_match,
    GOOD_VERDICTS as _GOOD_VERDICTS,
    _resolve_mismatch_via_search,
)

# CH helpers (optional — for UK Companies House verification)
try:
    from ch_helpers import ch_get, ch_search, ch_verify_exact, init_ch_keys
    init_ch_keys([
        "b8cfc466-eed0-4645-92b0-0b2880e3fa02",
        "c82029cc-ed04-404b-8025-6725a4fffb35",
        "51f44e97-79e3-4c09-b613-454ff0557d16",
        "53fa17b8-fc81-4cb9-b613-454ff0557d16",
        "17dda79d-f01b-4a2e-917e-c516fda90177",
        "c56a7b53-88aa-41fd-9069-b70eaf6b27ce",
        "5bed5e2e-fa38-41d6-a7f6-ede520c5ddfd",
        "5ade06a5-88c3-46ea-b13b-5b00d3633a71",
        "aeafc247-7dc9-490a-8f7f-b952b4201a6c",
    ])
    HAS_CH = True
except Exception:
    HAS_CH = False

# ---------------------------------------------------------------------------
# NOTE: no global search rate-limit needed — domain.py's _ddgs_library_search
# already has per-engine cooldown on 429/403 (yandex/ddg/bing rotation).
# The standalone scripts (accuracy_check_us_le_domain.py) run with no global
# limit and outperform; we match that here.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
WORKERS = 6


def _get_drive_sa():
    """Load Google SA credentials.

    Resolution order:
      1. Streamlit secrets       (GOOGLE_SA_JSON table)            — Cloud / prod
      2. Env var                 (GOOGLE_SA_JSON = full JSON text) — CI / scripted
      3. Env var path            (GOOGLE_SA_JSON_PATH = file path) — explicit local
      4. ../credentials/*.json   (auto-pick newest)                — local dev default
    """
    if "GOOGLE_SA_JSON" in st.secrets:
        sa = dict(st.secrets["GOOGLE_SA_JSON"])
        if "private_key" in sa:
            sa["private_key"] = sa["private_key"].replace("\\n", "\n")
        return sa
    if "GOOGLE_SA_JSON" in os.environ:
        return _json.loads(os.environ["GOOGLE_SA_JSON"])
    if "GOOGLE_SA_JSON_PATH" in os.environ:
        with open(os.environ["GOOGLE_SA_JSON_PATH"]) as f:
            return _json.load(f)

    # Local-dev fallback: auto-pick the newest *.json in ../credentials/
    here = os.path.dirname(os.path.abspath(__file__))
    cred_dir = os.path.join(os.path.dirname(here), "credentials")
    if os.path.isdir(cred_dir):
        candidates = [os.path.join(cred_dir, f) for f in os.listdir(cred_dir) if f.endswith(".json")]
        candidates = [p for p in candidates if os.path.isfile(p)]
        if candidates:
            path = max(candidates, key=os.path.getmtime)
            with open(path) as f:
                sa = _json.load(f)
            st.sidebar.caption(f"SA loaded from `credentials/{os.path.basename(path)}`")
            return sa

    st.error("No service account found. Set GOOGLE_SA_JSON in .streamlit/secrets.toml, "
             "or set GOOGLE_SA_JSON_PATH env var, or drop a *.json into ../credentials/")
    st.stop()


# ---------------------------------------------------------------------------
# PLAYWRIGHT FETCH (JS-rendered sites) — graceful degradation on cloud
# ---------------------------------------------------------------------------
def _domain_alive(domain: str, timeout=4) -> bool:
    """Cheap pre-check: DNS + HTTP HEAD. Skips Playwright on dead domains."""
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
    except Exception:
        return False
    try:
        import requests
        r = requests.head(f"https://{domain}", timeout=timeout, allow_redirects=True)
        if r.status_code < 500:
            return True
    except Exception:
        pass
    try:
        import requests
        r = requests.head(f"http://{domain}", timeout=timeout, allow_redirects=True)
        return r.status_code < 500
    except Exception:
        return False


def playwright_fetch_domain_pages(domain):
    """Playwright fallback for JS-rendered sites.

    Pre-checks DNS+HEAD to skip dead domains (saves ~80s per dead row).
    Caps at homepage + 1 priority path with tight timeouts.
    """
    if not _domain_alive(domain):
        return ''
    try:
        from playwright.sync_api import sync_playwright
        from urllib.parse import urljoin, urlparse
    except ImportError:
        return ''

    base = f"https://{domain}"
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
                    if any(kw in text for kw in ['privacy', 'legal', 'terms', 'about us', 'about']):
                        if href and href.startswith('http'):
                            discovered_links.append(href)
                    elif any(kw in href.lower() for kw in ['privacy', 'legal', 'terms', 'about']):
                        if href and href.startswith('http'):
                            discovered_links.append(href)
                text = page.inner_text('body')
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    collected.append(f"=== / (playwright) ===\n{text[:4000]}")
            except Exception:
                pass

            all_urls = list(dict.fromkeys(discovered_links[:3]))
            for path in ['/privacy-policy', '/privacy', '/about']:
                url = urljoin(base, path)
                if url not in all_urls and url != base:
                    all_urls.append(url)

            for url in all_urls:
                if len(collected) >= 2:   # homepage + 1 priority page is enough
                    break
                try:
                    page.goto(url, timeout=8000, wait_until='domcontentloaded')
                    page.wait_for_timeout(1000)
                    text = page.inner_text('body')
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text and len(text) > 50:
                        label = urlparse(url).path or url
                        collected.append(f"=== {label} (playwright) ===\n{text[:4000]}")
                except Exception:
                    pass
            browser.close()
    except Exception:
        pass
    return "\n\n".join(collected)[:12000]


# ---------------------------------------------------------------------------
# CORE: Find LE from domain (3-step: forward search + reverse search + webfetch)
# ---------------------------------------------------------------------------
def _normalize_le_name(name):
    """Normalize LE name for comparison."""
    n = name.lower().strip()
    n = re.sub(r'\bltd\.?\b', 'limited', n)
    n = re.sub(r'\bp\.?l\.?c\.?\b', 'plc', n)
    n = re.sub(r'\bl\.?l\.?p\.?\b', 'llp', n)
    n = re.sub(r'\binc\.?\b', 'incorporated', n)
    n = re.sub(r'\bcorp\.?\b', 'corporation', n)
    n = re.sub(r'[^a-z0-9 ]', '', n)
    return re.sub(r'\s+', ' ', n).strip()


def _find_le_via_search(domain, search_type, query, country=''):
    """Web search → AI extraction to find LE for a domain."""
    results = web_search(query)
    if not results:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': f'{search_type}: no results', 'country': country}

    ctx = "\n".join(f"[{i+1}] URL: {r['url']}\n    Title: {r.get('title','')}\n    Snippet: {r.get('snippet','')}"
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
        text = ai_call(prompt, max_tokens=400)
        r = parse_json_safe(text)
        return {
            'le_name': r.get('le_name', '').strip(),
            'cin': str(r.get('cin', '')).strip(),
            'country': r.get('country', country).strip(),
            'confidence': r.get('confidence', 'low'),
            'reason': r.get('reason', ''),
        }
    except Exception as e:
        return {'le_name': '', 'cin': '', 'confidence': 'error',
                'reason': str(e), 'country': country}


def _find_le_via_webfetch(domain, country=''):
    """Fetch actual website pages → AI extract LE info."""
    page_text = fetch_domain_pages(domain)
    if not page_text or len(page_text.strip()) < 100:
        page_text = playwright_fetch_domain_pages(domain)
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
        text = ai_call(prompt, max_tokens=400)
        r = parse_json_safe(text)
        return {
            'le_name': r.get('le_name', '').strip(),
            'cin': str(r.get('cin', '')).strip(),
            'country': r.get('country', country).strip(),
            'confidence': r.get('confidence', 'low'),
            'reason': r.get('reason', ''),
        }
    except Exception as e:
        return {'le_name': '', 'cin': '', 'confidence': 'error',
                'reason': str(e), 'country': country}


def find_le_from_domain(domain, country=''):
    """
    3-step LE discovery (same flow as accuracy_check.py):
      1. Forward web search: "domain" company registration legal entity
      2. Reverse web search: site:domain "registered" OR "company number"
      3. Webfetch: actually visit the site pages → extract LE

    If 2/3 agree → high confidence. Webfetch is the authority.
    """
    country_hint = f' {country}' if country else ''

    # Step 1: Forward web search
    fwd = _find_le_via_search(domain, 'forward',
        f'"{domain}"{country_hint} company registration legal entity official', country)

    # Step 2: Reverse web search (different angle)
    rev = _find_le_via_search(domain, 'reverse',
        f'site:{domain} "registered in" OR "company number" OR "registered office"', country)

    # Step 3: Webfetch (the major call — actually visit the site)
    web = _find_le_via_webfetch(domain, country)

    # Merge results — if multiple agree, confidence is high
    candidates = [fwd, rev, web]
    found = [c for c in candidates if c.get('le_name')]

    if not found:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': 'No LE found from any method (forward/reverse/webfetch)',
                'country': country,
                'fwd_le': '', 'rev_le': '', 'web_le': ''}

    # Priority: webfetch > fwd+rev agreement > any single
    reason_parts = []

    # Check if webfetch found something — it's the authority
    if web.get('le_name'):
        best = web
        le_name = web['le_name']
        cin = web.get('cin', '')
        found_country = web.get('country', country)
        reason_parts.append(f"webfetch: {web.get('reason', '')}")

        # Check if fwd/rev agree with webfetch
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

        # Flag if fwd/rev found something different
        for label, other in [('forward', fwd), ('reverse', rev)]:
            if other.get('le_name') and _normalize_le_name(other['le_name']) != _normalize_le_name(le_name):
                reason_parts.append(f'{label} disagrees: {other["le_name"]}')

    else:
        # No webfetch — check if fwd and rev agree
        if fwd.get('le_name') and rev.get('le_name') and \
           _normalize_le_name(fwd['le_name']) == _normalize_le_name(rev['le_name']):
            # Both agree — pick the one with CIN
            best = fwd if fwd.get('cin') else rev
            le_name = best['le_name']
            cin = best.get('cin', '')
            found_country = best.get('country', country)
            confidence = 'medium'
            reason_parts.append('fwd+rev agree (no webfetch)')
        else:
            # Only one found, or they disagree — pick whichever has higher confidence
            candidates_sorted = sorted(found,
                key=lambda c: ('high', 'medium', 'low', 'none').index(c.get('confidence', 'none')))
            best = candidates_sorted[0]
            le_name = best['le_name']
            cin = best.get('cin', '')
            found_country = best.get('country', country)
            confidence = 'low'
            reason_parts.append(f"best single: {best.get('reason', '')}")
            # Flag disagreement
            for c in found:
                if _normalize_le_name(c['le_name']) != _normalize_le_name(le_name):
                    reason_parts.append(f'DISAGREEMENT: also found: {c["le_name"]}')

    return {
        'le_name': le_name,
        'cin': cin,
        'country': found_country,
        'confidence': confidence,
        'reason': ' | '.join(reason_parts),
        'fwd_le': fwd.get('le_name', ''),
        'rev_le': rev.get('le_name', ''),
        'web_le': web.get('le_name', ''),
    }


# ---------------------------------------------------------------------------
# CORE: Find Domain from LE name (LE → Domain mapping)
# ---------------------------------------------------------------------------
_AGGREGATOR_DOMAINS = {
    'linkedin.com', 'wikipedia.org', 'crunchbase.com', 'bloomberg.com',
    'zoominfo.com', 'opencorporates.com', 'dnb.com', 'rocketreach.co',
    'pitchbook.com', 'companieshouse.gov.uk', 'find-and-update.company-information.service.gov.uk',
    'sec.gov', 'glassdoor.com', 'indeed.com', 'facebook.com', 'twitter.com',
    'x.com', 'instagram.com', 'youtube.com', 'reddit.com', 'medium.com',
    'github.com', 'gov.uk', 'goo.gl', 'tracxn.com',
}


def _clean_domain(d: str) -> str:
    d = (d or '').strip().lower()
    d = re.sub(r'^https?://', '', d)
    d = re.sub(r'^www\.', '', d)
    return d.split('/')[0].split('?')[0].split('#')[0]


def find_domain_from_le(le_name, country='', cin=''):
    """
    Given an LE name (already a registered legal entity), find its OFFICIAL DOMAIN.

    Pipeline:
      1. Web search: "<le_name>" official website (and country / cin variants)
      2. AI extracts the official domain from results — rejects directories/aggregators
      3. Caller runs accuracy check on (LE, candidate_domain) to verify
    """
    queries = [
        f'"{le_name}" official website',
        f'"{le_name}" {country} official website' if country else None,
        f'"{le_name}" {cin} official website' if cin else None,
        f'"{le_name}" company homepage',
    ]
    queries = [q for q in queries if q]

    results = []
    for q in queries:
        results = web_search(q)
        if results:
            break

    if not results:
        return {'le_name': le_name, 'found_domain': '',
                'find_confidence': 'none',
                'find_reason': 'no search results from any query',
                'country': country, 'cin': cin}

    ctx = "\n".join(
        f"[{i+1}] URL: {r['url']}\n    Title: {r.get('title','')}\n    Snippet: {r.get('snippet','')}"
        for i, r in enumerate(results[:10]))

    country_hint = f' in {country}' if country else ''
    prompt = f"""Find the OFFICIAL website domain owned by this exact legal entity{country_hint}.

Entity Name:    "{le_name}"
Registration ID: {cin}
Country:        {country}

Search results:
{ctx}

Rules:
- OFFICIAL company website only — NOT LinkedIn, Wikipedia, Crunchbase, Bloomberg,
  ZoomInfo, OpenCorporates, Companies House, SEC, Glassdoor, social media,
  news articles, directories, or aggregators
- Must match THIS EXACT entity and country — not a similarly-named foreign entity
- Parked / for-sale / "domain available" pages = return empty ""
- If unsure or no clear official site, return empty "" — false positive is worse than a miss

Respond ONLY with JSON:
{{"domain": "example.com", "confidence": "high|medium|low", "reason": "brief"}}"""

    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=400))
        found = _clean_domain(r.get('domain', ''))
        # Reject obvious aggregators even if AI returned them
        if found in _AGGREGATOR_DOMAINS or any(found.endswith('.' + a) for a in _AGGREGATOR_DOMAINS):
            return {'le_name': le_name, 'found_domain': '',
                    'find_confidence': 'low',
                    'find_reason': f'AI returned aggregator domain "{found}" — rejected',
                    'country': country, 'cin': cin}
        return {
            'le_name': le_name,
            'found_domain': found,
            'find_confidence': r.get('confidence', 'low') if found else 'none',
            'find_reason': r.get('reason', ''),
            'country': country,
            'cin': cin,
        }
    except Exception as e:
        return {'le_name': le_name, 'found_domain': '',
                'find_confidence': 'error', 'find_reason': str(e),
                'country': country, 'cin': cin}


# ---------------------------------------------------------------------------
# CORE: Accuracy check
# ---------------------------------------------------------------------------
def run_accuracy_check_single(domain, le_name, country='', cin='', do_ch=False):
    """Run 3-check accuracy pipeline. Playwright retry if NO."""
    entry = {
        'entity_id': cin, 'country': country or '',
        'known_domain': domain, 'known_le_name': le_name,
    }
    try:
        with ThreadPoolExecutor(max_workers=3) as inner:
            f_fwd = inner.submit(check_forward, entry)
            f_rev = inner.submit(check_reverse, entry)
            f_web = inner.submit(check_webfetch, entry)
            fwd, rev, web = f_fwd.result(), f_rev.result(), f_web.result()
        merged = {**entry, **fwd, **rev, **web}
        final = compute_final_verdict(merged)
        result = {**merged, **final}

        # Playwright retry if NO
        if result.get('final_mapping_correct') == 'NO':
            pw_text = playwright_fetch_domain_pages(domain)
            if pw_text and pw_text.strip():
                web2 = _webfetch_with_text(entry, pw_text)
                old_v = web.get('webfetch_verdict', '')
                new_v = web2.get('webfetch_verdict', '')
                good = {'EXACT_MATCH', 'PARENT_MATCH', 'NAME_CHANGED', 'BRAND_NAME_MISMATCH'}
                if new_v in good or (old_v not in good and new_v != old_v):
                    merged2 = {**entry, **fwd, **rev, **web2}
                    final2 = compute_final_verdict(merged2)
                    result = {**merged2, **final2}

        # CH verification (only when requested + UK)
        ch_result = {}
        if do_ch and HAS_CH:
            if country and ('kingdom' in country.lower() or 'uk' in (country or '').upper()):
                ch_cin, ch_name, ch_st = ch_verify_exact(le_name, cin)
                ch_result = {'ch_le_name': ch_name, 'ch_cin': ch_cin, 'ch_status': ch_st}

        return {**result, **ch_result}
    except Exception as e:
        return {'final_mapping_correct': 'NEEDS_REVIEW', 'final_issue_notes': str(e)}


def _webfetch_with_text(entry, page_text):
    """Run webfetch check using pre-fetched Playwright text."""
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
  "POSSIBLE_MISMATCH"   : different name — relationship unclear
  "MISMATCH"            : clearly unrelated company
  "NOT_FOUND"           : not enough info

Respond ONLY with JSON:
{{"legal_name_on_site": "", "company_num_on_site": "", "parent_company": "",
  "verdict": "", "confidence": "high|medium|low", "explanation": ""}}"""

    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=800))
        result = {
            'webfetch_legal_name': r.get('legal_name_on_site', ''),
            'webfetch_company_num': r.get('company_num_on_site', ''),
            'webfetch_parent': r.get('parent_company', ''),
            'webfetch_verdict': r.get('verdict', 'NOT_FOUND'),
            'webfetch_confidence': r.get('confidence', 'low'),
            'webfetch_explanation': r.get('explanation', ''),
        }
        if result['webfetch_verdict'] in ('POSSIBLE_MISMATCH', 'MISMATCH'):
            site_name = result.get('webfetch_legal_name', '')
            if site_name and names_are_equivalent(site_name, kle):
                result['webfetch_verdict'] = 'EXACT_MATCH'
            elif site_name and fuzzy_name_match(site_name, kle) in _GOOD_VERDICTS:
                result['webfetch_verdict'] = fuzzy_name_match(site_name, kle)
            else:
                result = _resolve_mismatch_via_search(entry, result)
        return result
    except Exception as e:
        return {'webfetch_legal_name': '', 'webfetch_company_num': '',
                'webfetch_parent': '', 'webfetch_verdict': 'ERROR',
                'webfetch_confidence': '', 'webfetch_explanation': str(e)}


# ---------------------------------------------------------------------------
# GOOGLE SHEET I/O
# ---------------------------------------------------------------------------
_gspread_lock = threading.Lock()


def _get_gc():
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = _get_drive_sa()
    creds = Credentials.from_service_account_info(sa_info, scopes=[
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ])
    return gspread.authorize(creds)


def connect_gsheet(sheet_url):
    """Connect to Google Sheet and return (spreadsheet, worksheet, all_data)."""
    gc = _get_gc()
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', sheet_url)
    if not m:
        raise ValueError("Invalid Google Sheet URL")
    sheet_id = m.group(1)

    gid_match = re.search(r'gid=(\d+)', sheet_url)
    gid = int(gid_match.group(1)) if gid_match else None

    spreadsheet = gc.open_by_key(sheet_id)
    if gid is not None:
        ws = None
        for worksheet in spreadsheet.worksheets():
            if worksheet.id == gid:
                ws = worksheet
                break
        if not ws:
            ws = spreadsheet.sheet1
    else:
        ws = spreadsheet.sheet1

    all_data = ws.get_all_values()
    return spreadsheet, ws, all_data


def _idx_to_col(i):
    """Convert 1-indexed column number to letter(s)."""
    if i <= 26:
        return chr(64 + i)
    return chr(64 + (i - 1) // 26) + chr(65 + (i - 1) % 26)


def get_or_create_result_tab(spreadsheet, tab_name, headers, source_data=None, key_col='domain'):
    """Get or create a result tab. Returns worksheet + set of already-done keys.

    `key_col` is the header used as the resume key (e.g. 'domain' or 'le_name').
    """
    import gspread
    try:
        ws = spreadsheet.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        nrows = max(len(source_data) + 10 if source_data else 100, 100)
        ws = spreadsheet.add_worksheet(tab_name, rows=nrows, cols=len(headers) + 5)

    existing = ws.row_values(1)
    if not existing or existing[0] != headers[0]:
        ws.update('A1', [headers], value_input_option='RAW')
        time.sleep(0.5)

    # Load already-done keys for resume
    done_keys = set()
    all_data = ws.get_all_values()
    key_idx = headers.index(key_col) if key_col in headers else 0
    for row in all_data[1:]:
        if len(row) > key_idx and row[key_idx].strip():
            done_keys.add(row[key_idx].strip().lower())
    return ws, done_keys, len(all_data)


def write_batch_to_result_sheet(ws, rows, start_row, num_cols):
    """Write a batch of result rows to the result sheet. Thread-safe."""
    if not rows:
        return
    end_col = _idx_to_col(num_cols)
    range_str = f"A{start_row}:{end_col}{start_row + len(rows) - 1}"
    with _gspread_lock:
        for attempt in range(3):
            try:
                ws.update(range_str, rows, value_input_option='RAW')
                return
            except Exception as e:
                if '429' in str(e) or 'RATE_LIMIT' in str(e):
                    time.sleep(30 * (attempt + 1))
                elif attempt < 2:
                    time.sleep(5)
                else:
                    raise


# ---------------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Domain LE Console", page_icon="🔍", layout="wide")
st.title("Domain LE Console")

mode = st.sidebar.radio(
    "Mode",
    ["Map Domain → LE", "Map LE → Domain", "Check Accuracy"],
)
workers = WORKERS  # fixed at 6 — matches accuracy_check_us_le_domain.py

st.sidebar.markdown("---")
st.sidebar.markdown("**Input Source**")
input_source = st.sidebar.radio("Source", ["Upload File", "Google Sheet"])

# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------
df = None
gsheet_ws = None

if input_source == "Upload File":
    uploaded = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    if uploaded:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"Loaded {len(df)} rows")

elif input_source == "Google Sheet":
    sa_info = _get_drive_sa()
    sa_email = sa_info.get("client_email", "")
    sa_project = sa_info.get("project_id", "")
    if sa_email:
        st.info(
            f"Share your sheet with **{sa_email}** (Editor) before connecting.  \n"
            f"Project: `{sa_project}` — to swap, update `GOOGLE_SA_JSON` in "
            f"`.streamlit/secrets.toml` (Streamlit Cloud → App settings → Secrets)."
        )
    else:
        st.error("No service account loaded. Set GOOGLE_SA_JSON in .streamlit/secrets.toml")
    sheet_url = st.text_input("Google Sheet URL")
    sheet_tab = st.text_input("Tab name (leave empty for auto-detect from URL)")
    if sheet_url and st.button("Connect"):
        try:
            _, gsheet_ws, all_data = connect_gsheet(sheet_url)
            headers = all_data[0] if all_data else []
            data = all_data[1:] if len(all_data) > 1 else []
            df = pd.DataFrame(data, columns=headers)
            st.session_state['df'] = df
            st.session_state['gsheet_ws'] = gsheet_ws
            st.success(f"Connected! {len(df)} rows, tab: {gsheet_ws.title}")
        except Exception as e:
            import traceback
            err_type = type(e).__name__
            err_msg = str(e) or '(no message)'
            # gspread APIError carries response details on .response
            extra = ''
            resp = getattr(e, 'response', None)
            if resp is not None:
                try:
                    extra = f"\nHTTP {resp.status_code}: {resp.text[:500]}"
                except Exception:
                    pass
            st.error(f"**{err_type}**: {err_msg}{extra}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    if 'df' in st.session_state:
        df = st.session_state['df']
        gsheet_ws = st.session_state.get('gsheet_ws')

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Column mapping
    cols = list(df.columns)

    # -----------------------------------------------------------------------
    # MAP DOMAIN → LE  (domain in, legal entity out)
    # -----------------------------------------------------------------------
    if mode == "Map Domain → LE":
        st.subheader("Column Mapping")
        domain_col = st.selectbox("Domain column", cols, index=0)
        country_col = st.selectbox("Country column (optional)", ["-- None --"] + cols)
        if country_col == "-- None --":
            country_col = None

        default_country = st.text_input("Default country (e.g. 'United Kingdom', 'India')", "")
        verify_ch = st.checkbox("Verify on UK Companies House (for UK domains)", value=False)

        FIND_LE_HEADERS = [
            'domain', 'le_name', 'cin', 'country', 'confidence', 'reason',
            'fwd_le', 'rev_le', 'web_le',
            'forward_found_domain', 'forward_match', 'forward_confidence',
            'reverse_found_le', 'reverse_match', 'reverse_confidence',
            'webfetch_legal_name', 'webfetch_company_num', 'webfetch_verdict',
            'webfetch_confidence', 'webfetch_explanation',
            'final_mapping_correct', 'final_issue_notes',
        ]
        if verify_ch and HAS_CH:
            FIND_LE_HEADERS += ['ch_le_name', 'ch_cin', 'ch_status']
        BATCH_SIZE = 50

        if st.button("Run — Find LE", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_table = st.empty()

            # Setup result sheet if Google Sheet source
            result_ws = None
            done_domains = set()
            next_row = [2]
            spreadsheet = st.session_state.get('spreadsheet')

            if gsheet_ws:
                spreadsheet = gsheet_ws.spreadsheet
                result_ws, done_domains, existing_rows = get_or_create_result_tab(
                    spreadsheet, "Find LE + Accuracy", FIND_LE_HEADERS)
                next_row[0] = existing_rows + 1 if existing_rows > 1 else 2
                if done_domains:
                    st.info(f"Resuming — {len(done_domains)} domains already done")

            entries = []
            for idx, row in df.iterrows():
                domain = str(row[domain_col]).strip().lower()
                if not domain or domain == 'nan':
                    continue
                if domain in done_domains:
                    continue
                country = str(row[country_col]).strip() if country_col else default_country
                entries.append({'idx': idx, 'domain': domain, 'country': country})

            results = []
            pending_rows = []
            lock = threading.Lock()
            completed = [0]
            found = [0]

            def process_find_le(entry):
                result = find_le_from_domain(entry['domain'], entry['country'])
                accuracy_data = {}
                ch_data = {}

                # Run accuracy check if LE was found
                if result.get('le_name'):
                    acc = run_accuracy_check_single(
                        entry['domain'], result['le_name'],
                        result.get('country', entry['country']),
                        result.get('cin', ''),
                        do_ch=verify_ch)
                    accuracy_data = {
                        'forward_found_domain': acc.get('forward_found_domain', ''),
                        'forward_match': acc.get('forward_match', ''),
                        'forward_confidence': acc.get('forward_confidence', ''),
                        'reverse_found_le': acc.get('reverse_found_le', ''),
                        'reverse_match': acc.get('reverse_match', ''),
                        'reverse_confidence': acc.get('reverse_confidence', ''),
                        'webfetch_legal_name': acc.get('webfetch_legal_name', ''),
                        'webfetch_company_num': acc.get('webfetch_company_num', ''),
                        'webfetch_verdict': acc.get('webfetch_verdict', ''),
                        'webfetch_confidence': acc.get('webfetch_confidence', ''),
                        'webfetch_explanation': acc.get('webfetch_explanation', ''),
                        'final_mapping_correct': acc.get('final_mapping_correct', ''),
                        'final_issue_notes': acc.get('final_issue_notes', ''),
                    }
                    if verify_ch and acc.get('ch_le_name'):
                        ch_data = {
                            'ch_le_name': acc.get('ch_le_name', ''),
                            'ch_cin': acc.get('ch_cin', ''),
                            'ch_status': acc.get('ch_status', ''),
                        }

                return {**entry, **result, **accuracy_data, **ch_data}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_find_le, e): e for e in entries}
                for future in as_completed(futures):
                    try:
                        out = future.result()
                        with lock:
                            results.append(out)
                            completed[0] += 1
                            if out.get('le_name'):
                                found[0] += 1

                            # Build row for sheet
                            row_vals = [str(out.get(h, '')) for h in FIND_LE_HEADERS]
                            pending_rows.append(row_vals)

                            # Flush every BATCH_SIZE
                            if result_ws and len(pending_rows) >= BATCH_SIZE:
                                write_batch_to_result_sheet(
                                    result_ws, pending_rows, next_row[0], len(FIND_LE_HEADERS))
                                next_row[0] += len(pending_rows)
                                pending_rows.clear()
                                time.sleep(1)

                            pct = completed[0] / len(entries)
                            progress_bar.progress(pct)
                            status_text.text(f"Processed {completed[0]}/{len(entries)} — Found: {found[0]}")
                    except Exception:
                        with lock:
                            completed[0] += 1

            # Flush remaining
            if result_ws and pending_rows:
                write_batch_to_result_sheet(
                    result_ws, pending_rows, next_row[0], len(FIND_LE_HEADERS))
                pending_rows.clear()

            results_df = pd.DataFrame(results)
            results_df = results_df[[c for c in FIND_LE_HEADERS if c in results_df.columns]]
            st.session_state['results_df'] = results_df

            st.success(f"Done! Found LE for {found[0]}/{len(entries)} domains")
            if result_ws:
                sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
                st.success(f"Results written to **Find LE Results** tab → [Open Sheet]({sheet_url})")
            st.dataframe(results_df, use_container_width=True)

            csv = results_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "le_results.csv", "text/csv")

    # -----------------------------------------------------------------------
    # MAP LE → DOMAIN  (legal entity in, domain out, then verify)
    # -----------------------------------------------------------------------
    elif mode == "Map LE → Domain":
        st.subheader("Column Mapping")
        le_col = st.selectbox("LE Name column", cols, index=0)
        country_col = st.selectbox("Country column (optional)", ["-- None --"] + cols)
        cin_col = st.selectbox("Company Number column (optional)", ["-- None --"] + cols)
        if country_col == "-- None --":
            country_col = None
        if cin_col == "-- None --":
            cin_col = None

        default_country = st.text_input("Default country (e.g. 'United Kingdom', 'India')", "")
        verify_ch_lemap = st.checkbox("Verify on UK Companies House (for UK entities)", value=False)
        skip_verify = st.checkbox("Skip accuracy verification (faster, just find domain)", value=False)

        MAP_LE_HEADERS = [
            'le_name', 'country', 'cin',
            'found_domain', 'find_confidence', 'find_reason',
            'forward_found_domain', 'forward_match', 'forward_confidence',
            'reverse_found_le', 'reverse_match', 'reverse_confidence',
            'webfetch_legal_name', 'webfetch_company_num', 'webfetch_verdict',
            'webfetch_confidence', 'webfetch_explanation',
            'final_mapping_correct', 'final_issue_notes',
        ]
        if verify_ch_lemap and HAS_CH:
            MAP_LE_HEADERS += ['ch_le_name', 'ch_cin', 'ch_status']
        BATCH_SIZE = 50

        if st.button("Run — Map LE → Domain", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            result_ws = None
            done_keys = set()
            next_row = [2]
            spreadsheet = None

            if gsheet_ws:
                spreadsheet = gsheet_ws.spreadsheet
                result_ws, done_keys, existing_rows = get_or_create_result_tab(
                    spreadsheet, "LE → Domain Results", MAP_LE_HEADERS, key_col='le_name')
                next_row[0] = existing_rows + 1 if existing_rows > 1 else 2
                if done_keys:
                    st.info(f"Resuming — {len(done_keys)} LE names already done")

            entries = []
            for idx, row in df.iterrows():
                le_name = str(row[le_col]).strip()
                if not le_name or le_name.lower() == 'nan':
                    continue
                if le_name.lower() in done_keys:
                    continue
                country = str(row[country_col]).strip() if country_col else default_country
                cin = str(row[cin_col]).strip() if cin_col else ''
                entries.append({'idx': idx, 'le_name': le_name,
                                'country': country, 'cin': cin})

            results = []
            pending_rows = []
            lock = threading.Lock()
            completed = [0]
            found = [0]

            def process_map_le(entry):
                find_result = find_domain_from_le(
                    entry['le_name'], entry['country'], entry['cin'])
                accuracy_data = {}
                ch_data = {}

                if find_result.get('found_domain') and not skip_verify:
                    acc = run_accuracy_check_single(
                        find_result['found_domain'],
                        entry['le_name'],
                        entry['country'],
                        entry['cin'],
                        do_ch=verify_ch_lemap)
                    accuracy_data = {
                        'forward_found_domain': acc.get('forward_found_domain', ''),
                        'forward_match': acc.get('forward_match', ''),
                        'forward_confidence': acc.get('forward_confidence', ''),
                        'reverse_found_le': acc.get('reverse_found_le', ''),
                        'reverse_match': acc.get('reverse_match', ''),
                        'reverse_confidence': acc.get('reverse_confidence', ''),
                        'webfetch_legal_name': acc.get('webfetch_legal_name', ''),
                        'webfetch_company_num': acc.get('webfetch_company_num', ''),
                        'webfetch_verdict': acc.get('webfetch_verdict', ''),
                        'webfetch_confidence': acc.get('webfetch_confidence', ''),
                        'webfetch_explanation': acc.get('webfetch_explanation', ''),
                        'final_mapping_correct': acc.get('final_mapping_correct', ''),
                        'final_issue_notes': acc.get('final_issue_notes', ''),
                    }
                    if verify_ch_lemap and acc.get('ch_le_name'):
                        ch_data = {
                            'ch_le_name': acc.get('ch_le_name', ''),
                            'ch_cin': acc.get('ch_cin', ''),
                            'ch_status': acc.get('ch_status', ''),
                        }

                return {**entry, **find_result, **accuracy_data, **ch_data}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_map_le, e): e for e in entries}
                for future in as_completed(futures):
                    try:
                        out = future.result()
                        with lock:
                            results.append(out)
                            completed[0] += 1
                            if out.get('found_domain'):
                                found[0] += 1

                            row_vals = [str(out.get(h, '')) for h in MAP_LE_HEADERS]
                            pending_rows.append(row_vals)

                            if result_ws and len(pending_rows) >= BATCH_SIZE:
                                write_batch_to_result_sheet(
                                    result_ws, pending_rows, next_row[0], len(MAP_LE_HEADERS))
                                next_row[0] += len(pending_rows)
                                pending_rows.clear()
                                time.sleep(1)

                            pct = completed[0] / len(entries) if entries else 1
                            progress_bar.progress(pct)
                            status_text.text(
                                f"Processed {completed[0]}/{len(entries)} — "
                                f"Domain found: {found[0]}")
                    except Exception:
                        with lock:
                            completed[0] += 1

            if result_ws and pending_rows:
                write_batch_to_result_sheet(
                    result_ws, pending_rows, next_row[0], len(MAP_LE_HEADERS))
                pending_rows.clear()

            results_df = pd.DataFrame(results)
            results_df = results_df[[c for c in MAP_LE_HEADERS if c in results_df.columns]]
            st.session_state['results_df'] = results_df

            st.success(f"Done! Found domain for {found[0]}/{len(entries)} LE names")
            if result_ws:
                sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
                st.success(f"Results written to **LE → Domain Results** tab → [Open Sheet]({sheet_url})")
            st.dataframe(results_df, use_container_width=True)

            csv = results_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "le_to_domain_results.csv", "text/csv")

    # -----------------------------------------------------------------------
    # CHECK ACCURACY MODE
    # -----------------------------------------------------------------------
    elif mode == "Check Accuracy":
        st.subheader("Column Mapping")
        domain_col = st.selectbox("Domain column", cols, index=0)
        le_col = st.selectbox("LE Name column", cols, index=min(1, len(cols)-1))
        country_col = st.selectbox("Country column (optional)", ["-- None --"] + cols)
        cin_col = st.selectbox("Company Number column (optional)", ["-- None --"] + cols)
        if country_col == "-- None --":
            country_col = None
        if cin_col == "-- None --":
            cin_col = None

        default_country = st.text_input("Default country", "")
        verify_ch_acc = st.checkbox("Verify on UK Companies House", value=False)

        ACCURACY_HEADERS = [
            'domain', 'le_name', 'country',
            'forward_found_domain', 'forward_match', 'forward_confidence',
            'reverse_found_le', 'reverse_match', 'reverse_confidence',
            'webfetch_legal_name', 'webfetch_company_num', 'webfetch_parent',
            'webfetch_verdict', 'webfetch_confidence', 'webfetch_explanation',
            'final_mapping_correct', 'final_issue_notes',
        ]
        if verify_ch_acc and HAS_CH:
            ACCURACY_HEADERS += ['ch_le_name', 'ch_cin', 'ch_status']
        BATCH_SIZE = 50

        if st.button("Run — Check Accuracy", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Setup result sheet
            result_ws = None
            done_domains = set()
            next_row = [2]
            spreadsheet = None

            if gsheet_ws:
                spreadsheet = gsheet_ws.spreadsheet
                result_ws, done_domains, existing_rows = get_or_create_result_tab(
                    spreadsheet, "Accuracy Results", ACCURACY_HEADERS)
                next_row[0] = existing_rows + 1 if existing_rows > 1 else 2
                if done_domains:
                    st.info(f"Resuming — {len(done_domains)} domains already done")

            entries = []
            for idx, row in df.iterrows():
                domain = str(row[domain_col]).strip().lower()
                le_name = str(row[le_col]).strip()
                if not domain or not le_name or domain == 'nan' or le_name == 'nan':
                    continue
                if domain in done_domains:
                    continue
                country = str(row[country_col]).strip() if country_col else default_country
                cin = str(row[cin_col]).strip() if cin_col else ''
                entries.append({'idx': idx, 'domain': domain, 'le_name': le_name,
                                'country': country, 'cin': cin})

            results = []
            pending_rows = []
            lock = threading.Lock()
            completed = [0]
            counts = {'yes': 0, 'no': 0, 'review': 0}

            def process_accuracy(entry):
                return {**entry, **run_accuracy_check_single(
                    entry['domain'], entry['le_name'], entry['country'], entry['cin'],
                    do_ch=verify_ch_acc)}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_accuracy, e): e for e in entries}
                for future in as_completed(futures):
                    try:
                        out = future.result()
                        with lock:
                            results.append(out)
                            completed[0] += 1
                            v = out.get('final_mapping_correct', '')
                            if v == 'YES': counts['yes'] += 1
                            elif v == 'NO': counts['no'] += 1
                            else: counts['review'] += 1

                            row_vals = [str(out.get(h, '')) for h in ACCURACY_HEADERS]
                            pending_rows.append(row_vals)

                            if result_ws and len(pending_rows) >= BATCH_SIZE:
                                write_batch_to_result_sheet(
                                    result_ws, pending_rows, next_row[0], len(ACCURACY_HEADERS))
                                next_row[0] += len(pending_rows)
                                pending_rows.clear()
                                time.sleep(1)

                            pct = completed[0] / len(entries)
                            progress_bar.progress(pct)
                            status_text.text(
                                f"Processed {completed[0]}/{len(entries)} — "
                                f"YES: {counts['yes']} | NO: {counts['no']} | REVIEW: {counts['review']}")
                    except Exception:
                        with lock:
                            completed[0] += 1

            # Flush remaining
            if result_ws and pending_rows:
                write_batch_to_result_sheet(
                    result_ws, pending_rows, next_row[0], len(ACCURACY_HEADERS))
                pending_rows.clear()

            results_df = pd.DataFrame(results)
            results_df = results_df[[c for c in ACCURACY_HEADERS if c in results_df.columns]]
            st.session_state['results_df'] = results_df

            # Summary
            col1, col2, col3 = st.columns(3)
            col1.metric("YES", counts['yes'])
            col2.metric("NO", counts['no'])
            col3.metric("NEEDS REVIEW", counts['review'])

            if result_ws:
                sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
                st.success(f"Results written to **Accuracy Results** tab → [Open Sheet]({sheet_url})")
            st.dataframe(results_df, use_container_width=True)

            csv = results_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "accuracy_results.csv", "text/csv")
