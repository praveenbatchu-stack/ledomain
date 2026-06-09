#!/usr/bin/env python
"""
Run LE → Domain mapping + accuracy on legal_entities_5k.xlsx
  Input  columns : id, entityId, name
  Output         : le5k_results.xlsx  (resumable — skips already-done LE names)

Pipeline per row:
  1. find_domain_from_le(le_name, country, cin)
  2. accuracy_check(found_domain, le_name, country, cin)  — fwd+rev+webfetch verdict
"""
import os, re, sys, threading, time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

os.environ.setdefault("NVIDIA_API_KEY",
    "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")

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

def _resolve_input_output(input_file=None, output_file=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if input_file is None:
        input_file = os.path.join(base_dir, "..", "console/Tracxn - POC - Amazon.xlsx")
    if output_file is None:
        out_name = os.path.splitext(os.path.basename(input_file))[0] + "_results.xlsx"
        output_file = os.path.join(os.path.dirname(os.path.abspath(input_file)), out_name)
    return input_file, output_file

INPUT_FILE, OUTPUT_FILE = _resolve_input_output()
WORKERS     = int(os.environ.get("WORKERS", "6"))
SAVE_EVERY  = 20
DEFAULT_COUNTRY = os.environ.get("DEFAULT_COUNTRY", "United States")

PRIORITY_PATHS = ['/privacy-policy', '/privacy', '/terms', '/terms-of-use',
                  '/terms-of-service', '/legal', '/legal-notice']
EXTRA_PATHS    = ['/', '/about', '/about-us', '/company', '/contact']

_AGGREGATOR_DOMAINS = {
    'linkedin.com', 'wikipedia.org', 'crunchbase.com', 'bloomberg.com',
    'zoominfo.com', 'opencorporates.com', 'dnb.com', 'rocketreach.co',
    'pitchbook.com', 'companieshouse.gov.uk', 'find-and-update.company-information.service.gov.uk',
    'sec.gov', 'sunbiz.org', 'glassdoor.com', 'indeed.com', 'facebook.com',
    'twitter.com', 'x.com', 'instagram.com', 'youtube.com', 'reddit.com',
    'medium.com', 'github.com', 'gov.uk', 'goo.gl', 'tracxn.com',
    'bbb.org', 'yelp.com', 'manta.com', 'buzzfile.com',
}


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


def _domain_alive(domain: str, timeout=4) -> bool:
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
    except Exception:
        return False
    for scheme in ('https://', 'http://'):
        try:
            r = requests.head(f"{scheme}{domain}", timeout=timeout, allow_redirects=True)
            if r.status_code < 500:
                return True
        except Exception:
            pass
    return False


def fetch_all_pages(domain: str) -> str:
    if not _domain_alive(domain):
        return ''
    collected = []
    for path in PRIORITY_PATHS:
        url = urljoin(f'https://{domain}', path)
        text = _fetch_requests(url)
        if len(text.strip()) < 300:
            text = _fetch_playwright_page(domain, path)
        if len(text.strip()) > 100:
            collected.append(f'=== {path} ===\n{text}')
            break
    for path in EXTRA_PATHS:
        if len(collected) >= 4:
            break
        url = urljoin(f'https://{domain}', path)
        text = _fetch_requests(url)
        if len(text.strip()) < 300:
            text = _fetch_playwright_page(domain, path)
        if len(text.strip()) > 100:
            collected.append(f'=== {path} ===\n{text}')
    if not collected:
        text = _fetch_playwright_page(domain, '/')
        if text:
            collected.append(f'=== / (browser) ===\n{text}')
    return '\n\n'.join(collected)[:12000]


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


# ---------------------------------------------------------------------------
# LE → Domain finder (multi-query merge → AI extract official domain)
#
# Adapted from domain_search_priority.py forward_search:
#   - strips legal suffix so search matches the trade name
#   - MERGES results from multiple query variants (not break-on-first) — small
#     LLCs only show up once you drop the quotes / suffix
#   - permissive prompt: accept the company's own .com even if BBB/Sunbiz
#     dominate the results
# ---------------------------------------------------------------------------
def _clean_domain(d: str) -> str:
    d = (d or '').strip().lower()
    d = re.sub(r'^https?://', '', d)
    d = re.sub(r'^www\.', '', d)
    return d.split('/')[0].split('?')[0].split('#')[0]


_LEGAL_SUFFIX_RE = re.compile(
    r'\s*(?:'
    r'L\.?\s*L\.?\s*C\.?|LLC|'
    r'L\.?\s*L\.?\s*P\.?|LLP|'
    r'P\.?\s*L\.?\s*L\.?\s*C\.?|PLLC|'
    r'INC\.?(?:ORPORATED)?|CORP\.?(?:ORATION)?|'
    r'CO\.?(?:MPANY)?|COMPANY|'
    r'LTD\.?|LIMITED|PLC|'
    r'PVT\.?\s*LTD\.?|PRIVATE\s+LIMITED|'
    r'GMBH|S\.?A\.?|S\.?L\.?|S\.?R\.?L\.?|N\.?V\.?|B\.?V\.?|'
    r'PTY\.?\s*LTD\.?'
    r')\s*$',
    re.IGNORECASE,
)


def _strip_legal_suffix(name: str) -> str:
    prev = None
    out = name.strip()
    while out != prev:
        prev = out
        out = _LEGAL_SUFFIX_RE.sub('', out).strip().rstrip(',').rstrip('.').strip()
    return out


def _merge_search_results(*lists, limit=12):
    seen, merged = set(), []
    for lst in lists:
        for r in (lst or []):
            url = r.get('url', '')
            if url and url not in seen:
                seen.add(url)
                merged.append(r)
                if len(merged) >= limit:
                    return merged
    return merged


def find_domain_from_le(le_name, country='', cin=''):
    short = _strip_legal_suffix(le_name)
    country_q = country or ''

    queries = [
        f'"{le_name}" official website',
        f'{short} {country_q} official website'.strip(),
        f'{short} company website',
        f'"{short}" official site',
    ]
    if cin:
        queries.append(f'"{le_name}" {cin}')
    # dedupe while preserving order
    _seen = set()
    queries = [q for q in queries
               if q and (q not in _seen and not _seen.add(q))]

    all_results = []
    for q in queries:
        r = web_search(q)
        if r:
            all_results.append(r)
            # stop early once we have enough merged hits
            if len(_merge_search_results(*all_results)) >= 8:
                break

    results = _merge_search_results(*all_results)
    if not results:
        return {'found_domain': '', 'find_confidence': 'none',
                'find_reason': 'no search results from any query'}

    ctx = "\n".join(
        f"[{i+1}] URL: {r['url']}\n    Title: {r.get('title','')}\n    Snippet: {r.get('snippet','')}"
        for i, r in enumerate(results[:10]))

    country_hint = f' in {country}' if country else ''
    prompt = f"""Find the OFFICIAL website domain owned by this legal entity{country_hint}.

Entity Name:    "{le_name}"
Trade name:     "{short}"
Registration:   {cin}
Country:        {country}

Search results:
{ctx}

GOAL: Identify the company's own website. Many small US LLCs have basic
.com / .net / .us / .biz sites — pick them when the host clearly corresponds
to the entity. Confidence should be HIGH when the host clearly matches the
trade name; MEDIUM when partial; LOW or empty if uncertain.

ACCEPT — pick the domain in cases like these:
  "DELUXE ATHLETICS, LLC"     → deluxeathletics.com    (host = both words)
  "TROON GOLF, L.L.C."        → troongolf.com          (host = both words)
  "ECHLER SECURITY LLC"       → echlersecurity.com     (both meaningful words)
  "SAVAGE AVIATION LLC"       → savageaviationllc.com  (host echoes entity)
  "SELAH PHOTOGRAPHY LLC"     → selahphotographer.com  (close stem match)
  "ARC ADVISORY LLC"          → arc-advisorsllc.com    (both tokens, plural OK)
  "ARC ADVISORY LLC"          → growwitharc.com        (Arc Advisors brand)
  "APPLE INC"                 → apple.com              (single distinctive word)
  "BAZ LLC"                   → baz.llc                (entity-named domain)

REJECT — return empty "" in cases like these (the #1 failure mode):
  Domains that share just ONE generic word with a MULTI-WORD entity name and
  are obviously a different unrelated company. Especially watch for hosts
  that look like "join{{X}} / try{{X}} / get{{X}} / use{{X}} / app{{X}} / my{{X}}"
  where {{X}} is just one common word from the entity:
    "ARC ADVISORY LLC"        vs joinarc.com  → REJECT  (joinarc.com is the
                                                          fintech "Arc, Inc." —
                                                          a different company.
                                                          Word "advisory" missing
                                                          from host.)
    "STAR HOLDINGS LLC"       vs trystar.com  → REJECT  (different company)
    "PRIME REALTY GROUP"      vs getprime.com → REJECT  (different company)

  Also reject:
  - Profile / aggregator pages, NOT the company's own site:
    LinkedIn, Facebook, Instagram, Twitter/X, YouTube, TikTok, Wikipedia,
    Crunchbase, Bloomberg, ZoomInfo, OpenCorporates, BBB, Yelp, Manta,
    Akama, Sunbiz, SEC, gov registries, news articles, Reddit posts.
  - Parked / for-sale / "domain available" pages.
  - Domains that match only one generic word and the rest of the entity name
    is missing entirely from the host (no shared meaning).

DECISION:
- If a candidate clearly is the entity's own site → pick it (high/medium).
- If no candidate clearly belongs to the entity → return empty "".

Respond ONLY with JSON:
{{"domain": "example.com", "confidence": "high|medium|low", "reason": "brief"}}"""

    try:
        r = parse_json_safe(ai_call(prompt, max_tokens=400))
        found = _clean_domain(r.get('domain', ''))
        if found in _AGGREGATOR_DOMAINS or any(found.endswith('.' + a) for a in _AGGREGATOR_DOMAINS):
            return {'found_domain': '', 'find_confidence': 'low',
                    'find_reason': f'AI returned aggregator "{found}" — rejected'}
        return {
            'found_domain': found,
            'find_confidence': r.get('confidence', 'low') if found else 'none',
            'find_reason': r.get('reason', ''),
        }
    except Exception as e:
        return {'found_domain': '', 'find_confidence': 'error', 'find_reason': str(e)}




# ---------------------------------------------------------------------------
# Accuracy check on (le_name, found_domain)
# ---------------------------------------------------------------------------
def _webfetch_accuracy(entry, page_text):
    kd, kle = entry['known_domain'], entry['known_le_name']
    eid, cty = entry.get('entity_id', ''), entry.get('country', '')
    if not page_text.strip():
        return {'webfetch_legal_name': '', 'webfetch_company_num': '',
                'webfetch_parent': '', 'webfetch_verdict': 'FETCH_FAILED',
                'webfetch_confidence': '', 'webfetch_explanation': ''}
    prompt = f"""You fetched pages from "{kd}". Decide whether this domain belongs
to the legal entity below.

EXPECTED ENTITY:
  Name:           {kle}
  Registration:   {eid}
  Country:        {cty}

WEBSITE CONTENT (footers, privacy, about, terms):
{page_text[:7000]}

──── PICK EXACTLY ONE VERDICT ────

EXACT_MATCH       — the legal name shown on the site matches "{kle}" (same
                    company; minor differences like "LLC" vs "L.L.C." or
                    punctuation are fine, but it must be the same name).

NAME_CHANGED      — site is the same entity that has been renamed. Requires
                    an explicit on-site statement: "formerly known as {kle}",
                    "previously {kle}", "now operating as <new name>", or
                    very widely documented public knowledge (Facebook → Meta).
                    Otherwise → MISMATCH.

PARENT_MATCH      — domain owned by parent / holding / sister of {kle}.
                    Requires the site to EXPLICITLY identify the relationship,
                    e.g. "{kle} is a subsidiary of <site company>", "<site
                    company> is the parent of {kle}", "a {kle} portfolio
                    company". Otherwise → MISMATCH.

BRAND_NAME_MISMATCH — data has a brand / d/b/a name and the site shows the
                    registered legal name OF THE SAME COMPANY. Requires the
                    site to MENTION the brand name "{kle}" somewhere
                    (footer, "doing business as", brand list). Otherwise →
                    MISMATCH.

POSSIBLE_MISMATCH — site company name differs from "{kle}" but the domain
                    HOST clearly echoes the entity name (multiple distinctive
                    tokens of "{kle}" appear in the host, e.g.
                    "echlersecurity.com" for "ECHLER SECURITY &
                    INVESTIGATIONS L.L.C." showing "Echler USA"). The site
                    likely IS the same business operating under a slightly
                    different name, but there's no explicit on-site
                    confirmation. Send to manual review.

MISMATCH          — site is OBVIOUSLY a different company. Use only when:
                    (a) the on-site name has NO distinctive overlap with
                        "{kle}" (e.g. site is "Starbucks" for "STAR
                        HOLDINGS LLC"), OR
                    (b) the host is generic/startup-prefixed and shares only
                        ONE common word with "{kle}" (e.g. "joinarc.com"
                        showing "Arc Technologies, Inc" for "ARC ADVISORY
                        LLC" — different industries, only "Arc" overlaps).

NOT_FOUND         — site content too thin to decide.

──── ANTI-HALLUCINATION RULES — READ EVERY TIME ────

The single biggest mistake is asserting a relationship (PARENT_MATCH /
NAME_CHANGED / BRAND_NAME_MISMATCH) just because the site company and the
expected entity share a common word. DO NOT DO THIS.

Two companies with similar names — even very similar names — are usually
DIFFERENT LEGAL ENTITIES, especially in the US where many small LLCs reuse
common words. Different state of registration, different EIN/CIN, different
address → different entities. The fact that both have "Arc Advisory" or
"Prime" or "Star" in their name is a COINCIDENCE, not evidence.

Concrete examples — choose MISMATCH for these:

  Expected: "ARC ADVISORY LLC" (Florida)
  Site:     "Arc Technologies, Inc" / "Arc Advisory Group, Inc" (Massachusetts)
  → MISMATCH. Two different companies that both contain "Arc Advisory".
    No on-site statement that one owns/rebrands/acquired the other.

  Expected: "PRIME REALTY GROUP LLC"
  Site:     "Prime Bank Inc"
  → MISMATCH. Different industries, only shared word is "Prime".

  Expected: "STAR HOLDINGS LLC"
  Site:     "Starbucks Corporation"
  → MISMATCH. Obviously unrelated.

To assign PARENT_MATCH / NAME_CHANGED / BRAND_NAME_MISMATCH, you must POINT
to a specific phrase in the page content that establishes the relationship.
If you cannot quote such a phrase from the WEBSITE CONTENT block above,
default to MISMATCH.

Inferring a relationship from company-name similarity alone is forbidden.

Respond ONLY with JSON:
{{"legal_name_on_site": "exact name as shown on site",
  "company_num_on_site": "or empty",
  "parent_company": "only if explicitly stated on the site, else empty",
  "verdict": "EXACT_MATCH | NAME_CHANGED | PARENT_MATCH | BRAND_NAME_MISMATCH | POSSIBLE_MISMATCH | MISMATCH | NOT_FOUND",
  "confidence": "high|medium|low",
  "explanation": "one or two sentences citing what the site actually says"}}"""
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
    entry = {'entity_id': cin, 'country': country or '',
             'known_domain': domain, 'known_le_name': le_name}
    try:
        with ThreadPoolExecutor(max_workers=2) as inner:
            f_fwd = inner.submit(check_forward, entry)
            f_rev = inner.submit(check_reverse, entry)
            fwd, rev = f_fwd.result(), f_rev.result()
        page_text = _cached_fetch(domain)
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
    'id', 'entityId', 'name', 'country',
    'found_domain', 'find_confidence', 'find_reason',
    'forward_found_domain', 'forward_match', 'forward_confidence',
    'reverse_found_le', 'reverse_match', 'reverse_confidence',
    'webfetch_legal_name', 'webfetch_company_num', 'webfetch_verdict',
    'webfetch_confidence', 'webfetch_explanation',
    'final_mapping_correct', 'final_issue_notes',
]

def _process_one(entry):
    le_name, country, cin = entry['name'], entry['country'], entry['entityId']
    t0 = time.time()
    fr = find_domain_from_le(le_name, country, cin)
    domain = fr.get('found_domain', '')

    acc_out = {}
    if domain:
        a = accuracy_check(domain, le_name, country, cin)
        acc_out = {
            'forward_found_domain': a.get('forward_found_domain', ''),
            'forward_match':        a.get('forward_match', ''),
            'forward_confidence':   a.get('forward_confidence', ''),
            'reverse_found_le':     a.get('reverse_found_le', ''),
            'reverse_match':        a.get('reverse_match', ''),
            'reverse_confidence':   a.get('reverse_confidence', ''),
            'webfetch_legal_name':  a.get('webfetch_legal_name', ''),
            'webfetch_company_num': a.get('webfetch_company_num', ''),
            'webfetch_verdict':     a.get('webfetch_verdict', ''),
            'webfetch_confidence':  a.get('webfetch_confidence', ''),
            'webfetch_explanation': a.get('webfetch_explanation', ''),
            'final_mapping_correct': a.get('final_mapping_correct', ''),
            'final_issue_notes':    a.get('final_issue_notes', ''),
        }
        with _PAGE_CACHE_LOCK:
            _PAGE_CACHE.pop(domain, None)

    out = {**entry, **fr, **acc_out}
    out['_elapsed'] = f"{time.time()-t0:.1f}s"
    return out


def main(input_file=None, output_file=None, name_col='name', id_col='id', entity_col='entityId'):
    input_file, output_file = _resolve_input_output(input_file, output_file)
    df_in = pd.read_excel(input_file)
    df_in.columns = [c.strip() for c in df_in.columns]

    done = set()
    prev_results = []
    if os.path.exists(output_file):
        try:
            df_done = pd.read_excel(output_file)
            fd = df_done.get('found_domain', '').astype(str).str.strip()
            real_mask = (fd != '') & (fd.str.lower() != 'nan')

            wv = df_done.get('webfetch_verdict', '').astype(str).str.strip()
            fm = df_done.get('forward_match', '').astype(str).str.strip()
            rm = df_done.get('reverse_match', '').astype(str).str.strip()
            relationship_verdict = wv.isin(['PARENT_MATCH', 'ACQUISITION_MATCH',
                                            'BRAND_NAME_MISMATCH'])
            no_search_corroboration = ~fm.isin(['EXACT_MATCH', 'SUBDOMAIN_MATCH']) & \
                                        ~rm.isin(['EXACT_MATCH', 'CLOSE_MATCH', 'PARTIAL_MATCH'])
            quarantine_mask = relationship_verdict & no_search_corroboration

            keep = df_done[real_mask & ~quarantine_mask]
            done = set(keep['name'].dropna().astype(str).str.strip().str.lower())
            prev_results = keep.to_dict('records')

            n_quarantine = int(quarantine_mask.sum())
            n_empty = len(df_done) - int(real_mask.sum())
            print(f"Resuming — keep {len(done)} confirmed | "
                  f"retry {n_empty} previously-empty + {n_quarantine} quarantined "
                  f"(uncorroborated PARENT/ACQ/BRAND verdicts)")
        except Exception as e:
            print(f"Could not load previous results: {e}")

    entries = []
    for _, row in df_in.iterrows():
        le_name = str(row.get(name_col, '')).strip()
        if not le_name or le_name.lower() == 'nan':
            continue
        if le_name.lower() in done:
            continue
        entries.append({
            'id':       str(row.get(id_col, '')).strip(),
            'entityId': str(row.get(entity_col, '')).strip(),
            'name':     le_name,
            'country':  DEFAULT_COUNTRY,
        })

    print(f"LE names to process: {len(entries)} (workers={WORKERS}, country={DEFAULT_COUNTRY})")
    sys.stdout.flush()

    results = list(prev_results)
    lock = threading.Lock()
    completed = [0]
    found_dom = [0]
    yes_count = [0]

    def _flush():
        df_out = pd.DataFrame(results)
        for h in HEADERS:
            if h not in df_out.columns:
                df_out[h] = ''
        df_out[HEADERS].to_excel(output_file, index=False)

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(_process_one, e): e for e in entries}
            for future in as_completed(futures):
                try:
                    out = future.result()
                except Exception as ex_err:
                    out = {**futures[future], 'found_domain': '',
                           'final_mapping_correct': 'ERROR',
                           'final_issue_notes': str(ex_err)}
                with lock:
                    results.append(out)
                    completed[0] += 1
                    if out.get('found_domain'):
                        found_dom[0] += 1
                    if out.get('final_mapping_correct') == 'YES':
                        yes_count[0] += 1

                    verdict = out.get('final_mapping_correct', '') or '—'
                    dom_disp = out.get('found_domain') or '—'
                    print(f"[{completed[0]}/{len(entries)}] "
                          f"{out['name'][:45]:45s} → "
                          f"{dom_disp[:35]:35s} verdict={verdict:14s} "
                          f"[{out.get('_elapsed','')}] "
                          f"found={found_dom[0]} yes={yes_count[0]}",
                          flush=True)

                    if completed[0] % SAVE_EVERY == 0:
                        _flush()
                        print(f"  → saved {len(results)} rows to {output_file}",
                              flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — flushing partial results …", flush=True)

    _flush()
    print(f"\nDone! Found domain for {found_dom[0]}/{len(entries)} | YES verdict: {yes_count[0]}")
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run LE → Domain mapping')
    parser.add_argument('input_file', nargs='?', help='Input Excel file (default: legal_entities_5k.xlsx)')
    parser.add_argument('-o', '--output', help='Output Excel file (default: <input_basename>_results.xlsx)')
    parser.add_argument('--name-col', default='name', help='Column name for legal entity name')
    parser.add_argument('--id-col', default='id', help='Column name for row id')
    parser.add_argument('--entity-col', default='entityId', help='Column name for entity ID')
    args = parser.parse_args()
    main(args.input_file, args.output, args.name_col, args.id_col, args.entity_col)
