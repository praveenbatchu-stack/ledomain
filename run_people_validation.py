#!/usr/bin/env python
"""
Run LE validation on People-Apr-20-2026.xlsx
  Input : Domain Name, Geo, Stage
  Output: people_le_results.xlsx  (resumable — skips already-done domains)

Fetch strategy (same as accuracy_check_us_le_domain.py):
  1. requests (static) on priority pages (privacy/terms/legal)
  2. Playwright only on pages that return thin content (< 300 chars)
  3. Full Playwright homepage pass only if everything fails
"""
import os, re, threading, time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

os.environ.setdefault("NVIDIA_API_KEY",
    "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")

# Quiet noisy library loggers — keep our own prints visible
import logging
logging.getLogger().setLevel(logging.WARNING)
for name in ('httpx', 'urllib3', 'duckduckgo_search', 'ddgs', 'primp'):
    logging.getLogger(name).setLevel(logging.ERROR)

from domain import (
    check_forward, check_reverse, check_webfetch,
    compute_final_verdict, ai_call, parse_json_safe,
    web_search, HTTP_HEADERS,
    names_are_equivalent, fuzzy_name_match,
    GOOD_VERDICTS as _GOOD_VERDICTS,
    _resolve_mismatch_via_search,
)

INPUT_FILE  = "People-Apr-20-2026.xlsx"
OUTPUT_FILE = "people_le_results.xlsx"
WORKERS     = 6
SAVE_EVERY  = 20

PRIORITY_PATHS = ['/privacy-policy', '/privacy', '/terms', '/terms-of-use',
                  '/terms-of-service', '/legal', '/legal-notice']
EXTRA_PATHS    = ['/', '/about', '/about-us', '/company', '/contact']

# ---------------------------------------------------------------------------
# Hybrid fetch (requests-first, Playwright only when thin)
# ---------------------------------------------------------------------------
def _fetch_requests(url: str, max_chars=5000) -> str:
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=8, allow_redirects=True)
        if resp.status_code not in (200, 203):
            return ''
        html = resp.text
        html = re.sub(r'<(script|style|nav|header|footer|noscript)[^>]*>.*?</\1>',
                      '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception:
        return ''


def _fetch_playwright_page(domain: str, path: str, max_chars=5000) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return ''
    url = urljoin(f'https://{domain}', path)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled',
                      '--no-sandbox', '--disable-dev-shm-usage'],
            )
            ctx = browser.new_context(
                user_agent=HTTP_HEADERS['User-Agent'],
                viewport={'width': 1366, 'height': 800},
                locale='en-US',
            )
            page = ctx.new_page()
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=12000)
                try:
                    page.wait_for_load_state('networkidle', timeout=3000)
                except Exception:
                    pass
                body = page.evaluate('() => document.body ? document.body.innerText : ""') or ''
                body = re.sub(r'\s+', ' ', body).strip()
                return body[:max_chars] if len(body) > 80 else ''
            except Exception:
                return ''
            finally:
                browser.close()
    except Exception:
        return ''


def fetch_all_pages(domain: str) -> str:
    collected = []

    # Step 1: priority pages (privacy/legal) — requests, Playwright if thin
    for path in PRIORITY_PATHS:
        url = urljoin(f'https://{domain}', path)
        text = _fetch_requests(url)
        if len(text.strip()) < 300:
            text = _fetch_playwright_page(domain, path)
        if len(text.strip()) > 100:
            collected.append(f'=== {path} ===\n{text}')
            break

    # Step 2: extra pages (homepage/about)
    for path in EXTRA_PATHS:
        if len(collected) >= 4:
            break
        url = urljoin(f'https://{domain}', path)
        text = _fetch_requests(url)
        if len(text.strip()) < 300:
            text = _fetch_playwright_page(domain, path)
        if len(text.strip()) > 100:
            collected.append(f'=== {path} ===\n{text}')

    # Step 3: full Playwright fallback if nothing worked
    if not collected:
        text = _fetch_playwright_page(domain, '/')
        if text:
            collected.append(f'=== / (browser) ===\n{text}')
        text = _fetch_playwright_page(domain, '/privacy-policy')
        if text:
            collected.append(f'=== /privacy-policy (browser) ===\n{text}')

    return '\n\n'.join(collected)[:12000]


# ---------------------------------------------------------------------------
# LE finding
# ---------------------------------------------------------------------------
def _normalize(name):
    n = name.lower().strip()
    n = re.sub(r'\bltd\.?\b', 'limited', n)
    n = re.sub(r'\bp\.?l\.?c\.?\b', 'plc', n)
    n = re.sub(r'\bl\.?l\.?p\.?\b', 'llp', n)
    n = re.sub(r'\binc\.?\b', 'incorporated', n)
    n = re.sub(r'\bcorp\.?\b', 'corporation', n)
    n = re.sub(r'[^a-z0-9 ]', '', n)
    return re.sub(r'\s+', ' ', n).strip()


def _search_le(domain, search_type, query, country=''):
    results = web_search(query)
    if not results:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': f'{search_type}: no results', 'country': country}
    ctx = "\n".join(
        f"[{i+1}] URL: {r['url']}\n    Title: {r.get('title','')}\n    Snippet: {r.get('snippet','')}"
        for i, r in enumerate(results[:8]))
    country_hint = f' in {country}' if country else ''
    prompt = f"""Find the Legal Entity (LE) that OWNS and OPERATES the domain "{domain}"{country_hint}.

Search results ({search_type}):
{ctx}

TASK: Identify the REGISTERED LEGAL NAME of the company that owns "{domain}" and its registration number.
RULES: Return EXACT registered legal name with suffix. If unsure, return empty.

Respond ONLY with JSON:
{{"le_name": "EXACT REGISTERED LEGAL NAME", "cin": "company number or empty",
  "country": "country of registration or empty",
  "confidence": "high|medium|low", "reason": "brief"}}"""
    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=400))
        return {'le_name': r.get('le_name', '').strip(),
                'cin': str(r.get('cin', '')).strip(),
                'country': r.get('country', country).strip(),
                'confidence': r.get('confidence', 'low'),
                'reason': r.get('reason', '')}
    except Exception as e:
        return {'le_name': '', 'cin': '', 'confidence': 'error', 'reason': str(e), 'country': country}


_PAGE_CACHE = {}
_PAGE_CACHE_LOCK = threading.Lock()


def _cached_fetch(domain):
    with _PAGE_CACHE_LOCK:
        if domain in _PAGE_CACHE:
            return _PAGE_CACHE[domain]
    text = fetch_all_pages(domain)
    with _PAGE_CACHE_LOCK:
        _PAGE_CACHE[domain] = text
    return text


def _webfetch_le(domain, country=''):
    page_text = _cached_fetch(domain)
    if not page_text or len(page_text.strip()) < 50:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': 'could not fetch website', 'country': country}
    country_hint = f' in {country}' if country else ''
    prompt = f"""You fetched pages from "{domain}". Find the Legal Entity that owns this domain{country_hint}.

WEBSITE CONTENT:
{page_text[:7000]}

Look in footer, privacy policy, terms, about page for registered company name and number.
Return EXACT registered legal name with suffix.

Respond ONLY with JSON:
{{"le_name": "EXACT REGISTERED LEGAL NAME", "cin": "company number or empty",
  "country": "country of registration or empty",
  "confidence": "high|medium|low", "reason": "brief"}}"""
    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=400))
        return {'le_name': r.get('le_name', '').strip(),
                'cin': str(r.get('cin', '')).strip(),
                'country': r.get('country', country).strip(),
                'confidence': r.get('confidence', 'low'),
                'reason': r.get('reason', '')}
    except Exception as e:
        return {'le_name': '', 'cin': '', 'confidence': 'error', 'reason': str(e), 'country': country}


def find_le(domain, country=''):
    country_hint = f' {country}' if country else ''
    fwd = _search_le(domain, 'forward',
        f'"{domain}"{country_hint} company registration legal entity official', country)
    rev = _search_le(domain, 'reverse',
        f'site:{domain} "registered in" OR "company number" OR "registered office"', country)
    web = _webfetch_le(domain, country)

    found = [c for c in [fwd, rev, web] if c.get('le_name')]
    if not found:
        return {'le_name': '', 'cin': '', 'confidence': 'none',
                'reason': 'No LE found from any method',
                'country': country, 'fwd_le': '', 'rev_le': '', 'web_le': ''}

    if web.get('le_name'):
        le_name = web['le_name']
        agree = sum(1 for o in [fwd, rev]
                    if o.get('le_name') and _normalize(o['le_name']) == _normalize(le_name))
        confidence = 'high' if agree >= 1 else web.get('confidence', 'medium')
        reason = f"webfetch: {web.get('reason', '')} | {agree+1}/3 agree"
    elif fwd.get('le_name') and rev.get('le_name') and \
         _normalize(fwd['le_name']) == _normalize(rev['le_name']):
        best = fwd if fwd.get('cin') else rev
        le_name = best['le_name']
        confidence = 'medium'
        reason = 'fwd+rev agree (no webfetch)'
    else:
        best = sorted(found, key=lambda c: ('high','medium','low','none').index(
            c.get('confidence','none')))[0]
        le_name = best['le_name']
        confidence = 'low'
        reason = f"best single: {best.get('reason','')}"

    best_c = web if web.get('le_name') else (fwd if fwd.get('le_name') else rev)
    return {
        'le_name': le_name,
        'cin': best_c.get('cin', ''),
        'country': best_c.get('country', country),
        'confidence': confidence,
        'reason': reason,
        'fwd_le': fwd.get('le_name', ''),
        'rev_le': rev.get('le_name', ''),
        'web_le': web.get('le_name', ''),
    }


# ---------------------------------------------------------------------------
# Accuracy check
# ---------------------------------------------------------------------------
def _webfetch_accuracy(entry, page_text):
    kd, kle = entry['known_domain'], entry['known_le_name']
    eid, cty = entry.get('entity_id', ''), entry.get('country', '')
    if not page_text.strip():
        return {'webfetch_legal_name': '', 'webfetch_company_num': '',
                'webfetch_parent': '', 'webfetch_verdict': 'FETCH_FAILED',
                'webfetch_confidence': '', 'webfetch_explanation': ''}
    prompt = f"""You fetched pages from "{kd}". Verify if this domain belongs to "{kle}".

EXPECTED: Name={kle}  ID={eid}  Country={cty}

WEBSITE CONTENT:
{page_text[:7000]}

VERDICT: EXACT_MATCH | NAME_CHANGED | PARENT_MATCH | BRAND_NAME_MISMATCH | POSSIBLE_MISMATCH | MISMATCH | NOT_FOUND

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


def accuracy_check(domain, le_name, country='', cin=''):
    """Reuses cached page text from find_le; only does fwd+rev searches anew."""
    entry = {'entity_id': cin, 'country': country or '',
             'known_domain': domain, 'known_le_name': le_name}
    try:
        with ThreadPoolExecutor(max_workers=2) as inner:
            f_fwd = inner.submit(check_forward, entry)
            f_rev = inner.submit(check_reverse, entry)
            fwd, rev = f_fwd.result(), f_rev.result()
        page_text = _cached_fetch(domain)  # cached — no re-fetch
        web = _webfetch_accuracy(entry, page_text) if page_text.strip() \
              else check_webfetch(entry)
        merged = {**entry, **fwd, **rev, **web}
        final = compute_final_verdict(merged)
        return {**merged, **final}
    except Exception as e:
        return {'final_mapping_correct': 'NEEDS_REVIEW', 'final_issue_notes': str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
HEADERS = [
    'domain', 'geo', 'stage',
    'le_name', 'cin', 'country', 'confidence', 'reason',
    'fwd_le', 'rev_le', 'web_le',
    'forward_found_domain', 'forward_match', 'forward_confidence',
    'reverse_found_le', 'reverse_match', 'reverse_confidence',
    'webfetch_legal_name', 'webfetch_company_num', 'webfetch_verdict',
    'webfetch_confidence', 'webfetch_explanation',
    'final_mapping_correct', 'final_issue_notes',
]

df_in = pd.read_excel(INPUT_FILE)
df_in.columns = [c.strip() for c in df_in.columns]

done = set()
if os.path.exists(OUTPUT_FILE):
    try:
        df_done = pd.read_excel(OUTPUT_FILE)
        done = set(df_done['domain'].dropna().str.strip().str.lower())
        print(f"Resuming — {len(done)} already done")
    except Exception:
        pass

entries = []
for _, row in df_in.iterrows():
    domain = str(row.get('Domain Name', '')).strip().lower()
    if not domain or domain == 'nan' or domain in done:
        continue
    entries.append({
        'domain': domain,
        'geo': str(row.get('Geo', '')).strip(),
        'stage': str(row.get('Stage', '')).strip(),
    })

print(f"Domains to process: {len(entries)}")

results = []
lock = threading.Lock()
completed = [0]
found_count = [0]

def process(entry):
    """1. find_le (3-way LE discovery)  2. accuracy_check (verdict on the found pair)."""
    domain, geo = entry['domain'], entry['geo']
    t0 = time.time()
    le_result = find_le(domain, geo)

    acc = {}
    if le_result.get('le_name'):
        a = accuracy_check(domain, le_result['le_name'],
                           le_result.get('country', geo), le_result.get('cin', ''))
        acc = {
            'forward_found_domain': a.get('forward_found_domain', ''),
            'forward_match': a.get('forward_match', ''),
            'forward_confidence': a.get('forward_confidence', ''),
            'reverse_found_le': a.get('reverse_found_le', ''),
            'reverse_match': a.get('reverse_match', ''),
            'reverse_confidence': a.get('reverse_confidence', ''),
            'webfetch_legal_name': a.get('webfetch_legal_name', ''),
            'webfetch_company_num': a.get('webfetch_company_num', ''),
            'webfetch_verdict': a.get('webfetch_verdict', ''),
            'webfetch_confidence': a.get('webfetch_confidence', ''),
            'webfetch_explanation': a.get('webfetch_explanation', ''),
            'final_mapping_correct': a.get('final_mapping_correct', ''),
            'final_issue_notes': a.get('final_issue_notes', ''),
        }

    # Free cached page text now that this domain is done
    with _PAGE_CACHE_LOCK:
        _PAGE_CACHE.pop(domain, None)

    elapsed = time.time() - t0
    le_result['_elapsed'] = f"{elapsed:.1f}s"
    return {**entry, **le_result, **acc}

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(process, e): e for e in entries}
    for future in as_completed(futures):
        try:
            out = future.result()
        except Exception as ex_err:
            out = {**futures[future], 'le_name': '', 'final_mapping_correct': 'ERROR',
                   'final_issue_notes': str(ex_err)}
        with lock:
            results.append(out)
            completed[0] += 1
            if out.get('le_name'):
                found_count[0] += 1
                status = 'FOUND'
            else:
                status = 'NOT_FOUND'
            le_disp = out.get('le_name') or '—'
            verdict = out.get('final_mapping_correct', '') or '—'
            print(f"[{completed[0]}/{len(entries)}] {status:9s} {out['domain']:40s} → "
                  f"{le_disp[:55]:55s} ({out.get('confidence','5'):6s}) "
                  f"verdict={verdict:14s} [{out.get('_elapsed','')}] "
                  f"{out.get('reason','')[:60]}",
                  flush=True)

            if completed[0] % SAVE_EVERY == 0:
                df_out = pd.DataFrame(results)
                for h in HEADERS:
                    if h not in df_out.columns:
                        df_out[h] = ''
                df_out[HEADERS].to_excel(OUTPUT_FILE, index=False)
                print(f"  → saved {completed[0]} rows to {OUTPUT_FILE}")

# Final save
df_out = pd.DataFrame(results)
for h in HEADERS:
    if h not in df_out.columns:
        df_out[h] = ''
df_out[HEADERS].to_excel(OUTPUT_FILE, index=False)
print(f"\nDone! Found LE for {found_count[0]}/{len(entries)} domains")
print(f"Results saved to {OUTPUT_FILE}")
