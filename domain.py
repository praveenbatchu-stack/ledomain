#!/usr/bin/env python3
"""
Domain Accuracy Tester — Two-Way Verification + Website Deep-Check
====================================================================
Reads le.csv (entity_id, country, "domain - LE_NAME" format)
and performs THREE accuracy checks:

  Check 1 (Forward):   Search entity_id + LE name → find domain → compare with known domain
  Check 2 (Reverse):   Take known domain → find LE name → compare with known LE name
  Check 3 (WebFetch):  Actually visit known_domain (/, /about, /privacy-policy) →
                       extract real legal name, company number, parent company →
                       produce final verdict with nuanced match categories:

    EXACT_MATCH         — same entity, same name
    NAME_CHANGED        — same entity, company was renamed (e.g. Adobe Systems → Adobe Inc.)
    PARENT_MATCH        — domain belongs to parent company of the LE (mapping still valid)
    BRAND_NAME_MISMATCH — domain correct but LE name in data is brand name not legal name
                          (e.g. "Truecaller AB" vs legal "True Software Scandinavia AB")
    MISMATCH            — genuinely wrong domain or wrong entity
    NOT_FOUND           — could not determine

Resume support: adds only new entity_ids not already in accuracy_results.csv.
"""

import csv
import json
import os
import re
import sys
import time
import logging
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LE_CSV     = 'le.csv'
OUTPUT_CSV = 'accuracy_results.csv'

def _load_dotenv():
    """Zero-dependency .env loader (python-dotenv isn't installed in the
    anaconda env). Reads KEY=VALUE lines from console/.env and the project
    root .env into os.environ *without* clobbering vars already set by the
    caller. Quotes and inline whitespace around '=' are stripped."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, '.env'),
                 os.path.join(here, os.pardir, '.env')):
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except FileNotFoundError:
            continue


_load_dotenv()


def _load_nvidia_keys():
    """Build the NVIDIA NIM key pool. Sources, in priority order:
        1. NVIDIA_API_KEYS  — comma/newline-separated list (the rotation pool)
        2. NVIDIA_API_KEY   — single key, pushed to the front of the pool
    Falls back to the bundled pool below if the env has none. Order is
    preserved and duplicates removed."""
    keys = []
    single = os.environ.get("NVIDIA_API_KEY", "").strip()
    if single:
        keys.append(single)
    raw = os.environ.get("NVIDIA_API_KEYS", "")
    keys.extend(k.strip() for k in re.split(r"[,\n]", raw) if k.strip())
    if not keys:
        keys = list(_DEFAULT_NVIDIA_KEYS)
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# Fallback rotation pool used only when neither NVIDIA_API_KEY nor
# NVIDIA_API_KEYS is set in the environment / .env.
_DEFAULT_NVIDIA_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
]

# Pool of NVIDIA NIM keys round-robined across all ai_call() attempts so a
# single key's per-minute 429 isn't the bottleneck (mirrors FWS
# automate_jp.py's rotation over scrape_japan.NVIDIA_API_KEYS).
NVIDIA_API_KEYS = _load_nvidia_keys()
# Back-compat alias: first key in the pool. Some call-sites / main() still
# reference NVIDIA_API_KEY for the "is anything configured?" check.
NVIDIA_API_KEY = NVIDIA_API_KEYS[0] if NVIDIA_API_KEYS else ""

_NV_KEY_LOCK = threading.Lock()
_NV_KEY_IDX = [0]


def _next_nvidia_key():
    """Thread-safe round-robin over NVIDIA_API_KEYS. Each ai_call() attempt
    (and each concurrent worker) pulls the next key, so retries after a 429
    land on a different key instead of hammering the same one."""
    if not NVIDIA_API_KEYS:
        return ""
    with _NV_KEY_LOCK:
        idx = _NV_KEY_IDX[0] % len(NVIDIA_API_KEYS)
        _NV_KEY_IDX[0] += 1
        return NVIDIA_API_KEYS[idx]


# NVIDIA NIM text model — override with NVIDIA_TEXT_MODEL env var.
# Matches the LE domain console model configured for NVIDIA NIM calls.
TEXT_MODEL = os.environ.get("NVIDIA_TEXT_MODEL", "meta/llama-3.1-8b-instruct")

WORKERS    = 3
DELAY      = 1.2    # seconds between web searches
FETCH_TO   = 12     # seconds for website fetch timeout

# Ordered by how reliably each page carries the REGISTERED legal entity name.
# /contact, /privacy, /terms, /legal, /imprint typically disclose "<Brand> is
# operated by <Legal Co Pvt Ltd>, <address>" even when the homepage only shows
# the brand — so they come right after '/' and ahead of marketing /about pages.
FETCH_PATHS = ['/', '/contact', '/contact-us', '/privacy-policy', '/privacy',
               '/terms', '/legal', '/imprint', '/about', '/about-us', '/company']

HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

FIELDNAMES = [
    'entity_id', 'country', 'known_le_name', 'known_domain',
    'forward_found_domain', 'forward_match', 'forward_confidence', 'forward_reason',
    'reverse_found_le', 'reverse_match', 'reverse_confidence', 'reverse_reason',
    'webfetch_legal_name', 'webfetch_company_num', 'webfetch_parent',
    'webfetch_verdict', 'webfetch_confidence', 'webfetch_explanation',
    # ── Final verdict — WebFetch is primary source of truth ──────────
    'final_mapping_correct',  # YES / NO / NEEDS_REVIEW
    'final_issue_notes',      # plain-English: what is wrong / what to fix
]

GOOD_VERDICTS  = {'EXACT_MATCH', 'NAME_CHANGED', 'PARENT_MATCH', 'ACQUISITION_MATCH',
                  'SUBDOMAIN_MATCH', 'BRAND_NAME_MISMATCH', 'CLOSE_MATCH', 'PARTIAL_MATCH'}
BAD_VERDICTS   = {'MISMATCH'}
EMPTY_VERDICTS = {'NOT_FOUND', 'NO_RESULTS', 'FETCH_FAILED'}

STOP_WORDS = {'ltd','limited','inc','incorporated','plc','llp','llc','pvt',
              'private','public','publ','ab','asa','nv','bv','gmbh','ag','sa',
              'sas','srl','spa','oy','oyj','as','se','kk','corp','corporation',
              'the','co','company','group','holdings','holding','international',
              'technologies','technology','solutions','services','global',
              'systems','system','online','digital','software','media'}

# ---------------------------------------------------------------------------
# NAME NORMALIZATION — expand common abbreviations so trivial diffs don't
# trigger false mismatches  (e.g. "Pvt Ltd" ↔ "PRIVATE LIMITED")
# ---------------------------------------------------------------------------
ABBREVIATION_MAP = {
    'pvt': 'private', 'priv': 'private',
    'ltd': 'limited', 'ltda': 'limited',
    'inc': 'incorporated',
    'corp': 'corporation',
    'co': 'company',
    'intl': 'international', 'int': 'international',
    'tech': 'technology', 'techs': 'technologies',
    'svcs': 'services', 'svc': 'service',
    'sols': 'solutions', 'sol': 'solution',
    'sys': 'systems',
    'mfg': 'manufacturing',
    'engg': 'engineering', 'engr': 'engineering',
    'mgmt': 'management',
    'assoc': 'associates',
    'ind': 'industries', 'inds': 'industries',
    'grp': 'group',
    'hldgs': 'holdings', 'hlg': 'holding',
}


def normalize_company_name(name: str) -> str:
    """Normalize: lowercase, expand abbreviations, strip punctuation."""
    name = re.sub(r'[^a-z0-9\s]', ' ', name.lower().strip())
    words = name.split()
    return ' '.join(ABBREVIATION_MAP.get(w, w) for w in words)


def names_are_equivalent(a: str, b: str) -> bool:
    """Check if two names are the same after normalization."""
    if not a or not b:
        return False
    return normalize_company_name(a) == normalize_company_name(b)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WEB SEARCH — Yandex backend only via the ddgs library.
# DDG/Bing backends are heavily rate-limited (429/403) in our env, and the
# html.duckduckgo.com scraper returns 202 / no-results pages, so both have
# been dropped. Cooldown still kicks in on 429/403 from Yandex.
# ---------------------------------------------------------------------------
_YANDEX_COOLDOWN_UNTIL = 0.0
_engine_lock = threading.Lock()


def _yandex_is_cool() -> bool:
    with _engine_lock:
        return time.time() >= _YANDEX_COOLDOWN_UNTIL


def _yandex_cooldown(seconds: int):
    global _YANDEX_COOLDOWN_UNTIL
    until = time.time() + seconds
    with _engine_lock:
        _YANDEX_COOLDOWN_UNTIL = max(_YANDEX_COOLDOWN_UNTIL, until)


def _ddgs_library_search(query: str, max_results=8):
    """ddgs library pinned to Yandex backend (the only one returning 200s here)."""
    try:
        from ddgs import DDGS
    except Exception as e:
        log.debug(f"ddgs import failed: {e}")
        return []

    if not _yandex_is_cool():
        log.debug("skip yandex (cooling)")
        return []

    for attempt in range(2):
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results, backend='yandex'):
                    results.append({
                        'url': r.get('href', ''),
                        'title': r.get('title', ''),
                        'snippet': r.get('body', ''),
                    })
            return results
        except Exception as e:
            err = str(e).lower()
            if '429' in err or 'ratelimit' in err or '403' in err:
                _yandex_cooldown(60)
                log.warning(f"yandex blocked ({err[:60]}) — cooldown 60s")
                return []
            if 'timed out' in err or 'timeout' in err:
                time.sleep(1)
                continue
            log.debug(f"yandex error: {e}")
            return []
    return []


def web_search(query: str, retries=3):
    """Yandex-only search via ddgs library, with backoff retries."""
    for attempt in range(retries):
        r = _ddgs_library_search(query)
        if r:
            return r
        if attempt < retries - 1:
            wait = 2 * (attempt + 1)
            log.debug(f"web_search attempt {attempt+1} empty, retrying in {wait}s...")
            time.sleep(wait)
    return []


# ---------------------------------------------------------------------------
# WEBSITE FETCHER
# ---------------------------------------------------------------------------
def fetch_page(url: str, max_chars=4000):
    """Fetch a URL and return (raw_html, cleaned_text). Raw HTML is needed so
    callers can extract the page's own <a href> links (FWS-style discovery);
    cleaned text is the script/style-stripped, whitespace-collapsed body."""
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=FETCH_TO, allow_redirects=True)
        if resp.status_code >= 400:
            return '', ''
        html = resp.text
        stripped = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', ' ', html,
                          flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', stripped)
        text = re.sub(r'\s+', ' ', text).strip()
        return html, text[:max_chars]
    except Exception as e:
        log.debug(f"Fetch {url}: {e}")
        return '', ''


def fetch_page_text(url: str, max_chars=4000) -> str:
    return fetch_page(url, max_chars)[1]


# Page types most likely to disclose the REGISTERED legal entity, scored by how
# reliably each one carries "<Brand> is operated by <Legal Co Pvt Ltd>, <addr>".
# Used to rank the homepage's OWN links so a non-standard URL (/get-in-touch,
# /en/impressum, /company/legal-information) is still followed — unlike a fixed
# path list that only hits guessed slugs.
_PAGE_KEYWORDS = [
    (10, ('contact', 'get-in-touch', 'getintouch', 'reach-us', 'reachus', 'kontakt')),
    (9,  ('imprint', 'impressum', 'mentions-legales')),
    (8,  ('legal', 'disclaimer', 'disclosure', 'disclosures')),
    (7,  ('privacy', 'datenschutz')),
    (6,  ('terms', 'agb', 'conditions')),
    (5,  ('about', 'aboutus', 'who-we-are', 'company', 'corporate', 'overview')),
    (3,  ('team', 'people', 'leadership')),
]

_ANCHOR_RE = re.compile(
    r'<a\b[^>]*?href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL)


def _score_link(href: str, text: str) -> int:
    blob = f"{href} {text}".lower()
    return max((s for s, kws in _PAGE_KEYWORDS if any(k in blob for k in kws)),
               default=0)


def _discover_relevant_links(home_html: str, base: str, domain: str,
                             limit: int = 8) -> list:
    """Parse the homepage's own anchors and return the internal URLs most likely
    to disclose the registered legal entity (contact/imprint/legal/privacy/
    about), highest-confidence first. Mirrors FWS 3.0's link-summary crawl:
    follow the links the site actually exposes instead of guessing paths."""
    reg = domain.lower().lstrip('www.')
    scored: dict[str, int] = {}
    for m in _ANCHOR_RE.finditer(home_html or ''):
        href = m.group(1).strip()
        if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            continue
        text = re.sub(r'<[^>]+>', ' ', m.group(2))
        full = urljoin(base, href)
        host = urlparse(full).netloc.lower().lstrip('www.')
        if host and reg not in host and host not in reg:
            continue  # off-site link (social, partner, CDN)
        s = _score_link(href, text)
        if s <= 0:
            continue
        key = full.split('#')[0].rstrip('/')
        if s > scored.get(key, 0):
            scored[key] = s
    return [u for u, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:limit]


def fetch_domain_pages(domain: str, min_pages: int = 6) -> str:
    """Fetch the homepage, then follow the site's own legal/contact/about links
    (discovered from the homepage HTML) plus common fixed-path fallbacks, until
    at least `min_pages` pages are collected — so the registered legal entity on
    a /contact or /impressum page is seen even when the homepage shows only a
    brand."""
    collected = []
    for scheme in ('https', 'http'):
        base = f"{scheme}://{domain}"
        home_html, home_text = fetch_page(urljoin(base, '/'))
        if not home_text:
            continue
        collected.append(f"=== / ===\n{home_text}")
        seen = {urljoin(base, '/').split('#')[0].rstrip('/')}
        # Dynamically discovered links first, then fixed-path fallbacks for any
        # legal page the homepage didn't link to directly.
        candidates = _discover_relevant_links(home_html, base, domain)
        candidates += [urljoin(base, p) for p in FETCH_PATHS if p != '/']
        for url in candidates:
            key = url.split('#')[0].rstrip('/')
            if key in seen:
                continue
            seen.add(key)
            text = fetch_page_text(url)
            if text:
                label = urlparse(url).path or url
                collected.append(f"=== {label} ===\n{text}")
            if len(collected) >= min_pages:
                break
        if collected:
            break  # HTTPS worked — don't retry over HTTP
    return "\n\n".join(collected)[:10000]


# ---------------------------------------------------------------------------
# AI CALL
# ---------------------------------------------------------------------------
def ai_call(prompt: str, max_tokens: int = 700, max_retries: int = 5) -> str:
    if not NVIDIA_API_KEYS:
        raise RuntimeError("No NVIDIA API keys configured "
                           "(set NVIDIA_API_KEY or NVIDIA_API_KEYS).")
    # Give the rotation enough room to try every key at least once.
    max_retries = max(max_retries, len(NVIDIA_API_KEYS))
    for attempt in range(max_retries):
        key = _next_nvidia_key()
        try:
            resp = requests.post(
                'https://integrate.api.nvidia.com/v1/chat/completions',
                headers={'Content-Type': 'application/json',
                         'Authorization': f'Bearer {key}'},
                json={'model': TEXT_MODEL,
                      'messages': [
                          {'role': 'system', 'content': 'You are a precise entity verifier. Respond only with valid JSON.'},
                          {'role': 'user',   'content': prompt}
                      ],
                      'max_tokens': max_tokens, 'temperature': 0.1},
                timeout=90
            )
            resp.raise_for_status()
            data = resp.json()
            if 'error' in data:
                err_msg = str(data['error'])
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + 1
                    log.warning(f"AI call error (attempt {attempt+1}/{max_retries}): {err_msg} — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(err_msg)
            text = data['choices'][0]['message']['content'].strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'^```json\s*', '', text.strip())
            text = re.sub(r'\s*```$', '', text)
            return text.strip()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if attempt < max_retries - 1:
                if status_code == 429:
                    # We rotate to a fresh key on the next attempt, so don't
                    # sit out the full per-minute window — a short pause is
                    # enough. Honor Retry-After only if it's small.
                    ra = e.response.headers.get('Retry-After') if e.response is not None else None
                    wait = min(int(ra), 5) if (ra and ra.isdigit()) else 1
                    log.warning(f"HTTP 429 (attempt {attempt+1}/{max_retries}) — rotating key, retrying in {wait}s")
                else:
                    wait = 2 ** attempt + 1
                    log.warning(f"HTTP {status_code} (attempt {attempt+1}/{max_retries}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt + 1
                log.warning(f"Connection error (attempt {attempt+1}/{max_retries}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise


def parse_json_safe(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}


# ---------------------------------------------------------------------------
# FUZZY NAME MATCH
# ---------------------------------------------------------------------------
def core_words(s: str) -> set:
    norm = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', '', s.lower())).strip()
    return set(norm.split()) - STOP_WORDS


def fuzzy_name_match(a: str, b: str) -> str:
    na = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', '', a.lower())).strip()
    nb = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', '', b.lower())).strip()
    if na == nb:
        return 'EXACT_MATCH'
    if na in nb or nb in na:
        return 'PARTIAL_MATCH'
    ca, cb = core_words(a), core_words(b)
    if not ca or not cb:
        return 'MISMATCH'
    score = len(ca & cb) / max(len(ca), len(cb))
    if score >= 0.7:
        return 'CLOSE_MATCH'
    if score >= 0.4:
        return 'PARTIAL_MATCH'
    return 'MISMATCH'


# ---------------------------------------------------------------------------
# PARSE le.csv
# ---------------------------------------------------------------------------
def parse_le_csv(path: str):
    entries = []
    with open(path, newline='', encoding='utf-8') as f:
        sample = f.read(2048); f.seek(0)
        delimiter = '\t' if '\t' in sample else ','
        for row in csv.reader(f, delimiter=delimiter):
            if not row or len(row) < 3:
                continue
            entity_id = row[0].strip()
            country   = row[1].strip()
            raw       = row[2].strip()
            if ' - ' in raw:
                domain, le_name = raw.split(' - ', 1)
                domain  = domain.strip().lower().replace('www.', '')
                le_name = le_name.strip()
            else:
                domain  = raw.strip().lower()
                le_name = ''
            if entity_id:
                entries.append({'entity_id': entity_id, 'country': country,
                                'known_domain': domain, 'known_le_name': le_name})
    return entries


# ---------------------------------------------------------------------------
# CHECK 1: FORWARD
# ---------------------------------------------------------------------------
def check_forward(entry: dict) -> dict:
    le_name = entry['known_le_name']; eid = entry['entity_id']
    country = entry['country'];       kd  = entry['known_domain']

    results = web_search(f'"{le_name}" {eid} official website') or \
              web_search(f'"{le_name}" {country} official website')
    if not results:
        return {'forward_found_domain': '', 'forward_match': 'NO_RESULTS',
                'forward_confidence': '', 'forward_reason': 'no search results'}

    ctx = "\n".join(f"[{i+1}] URL: {r['url']}\n    Snippet: {r.get('snippet','')}"
                    for i, r in enumerate(results[:8]))
    prompt = f"""Find the OFFICIAL website domain for this exact legal entity:
Entity Name: "{le_name}"
Registration ID: {eid}
Country: {country}

Search results:
{ctx}

Rules:
- OFFICIAL company website only (not LinkedIn, Wikipedia, aggregators, directories)
- Must match THIS EXACT entity and country — not a similarly named foreign entity
- Parked / for-sale pages = return empty ""
- If unsure, return empty ""

Respond ONLY with JSON:
{{"domain": "example.com", "confidence": "high|medium|low", "reason": "brief reason"}}"""

    try:
        r = parse_json_safe(ai_call(prompt))
        found = r.get('domain', '').strip().lower()
        found = re.sub(r'^https?://', '', found)
        found = re.sub(r'^www\.', '', found).split('/')[0]
        conf  = r.get('confidence', 'low')
        rsn   = r.get('reason', '')
        if not found:
            match = 'NOT_FOUND'
        elif found == kd:
            match = 'EXACT_MATCH'
        elif found.endswith('.' + kd) or kd.endswith('.' + found):
            match = 'SUBDOMAIN_MATCH'
        else:
            match = 'MISMATCH'
        return {'forward_found_domain': found, 'forward_match': match,
                'forward_confidence': conf, 'forward_reason': rsn}
    except Exception as e:
        return {'forward_found_domain': '', 'forward_match': 'ERROR',
                'forward_confidence': '', 'forward_reason': str(e)}


# ---------------------------------------------------------------------------
# CHECK 2: REVERSE
# ---------------------------------------------------------------------------
def check_reverse(entry: dict) -> dict:
    kd  = entry['known_domain'];    kle = entry['known_le_name']
    cty = entry['country']

    results = web_search(f'"{kd}" company legal name registration {cty}') or \
              web_search(f'site:{kd} legal entity name')
    if not results:
        return {'reverse_found_le': '', 'reverse_match': 'NO_RESULTS',
                'reverse_confidence': '', 'reverse_reason': 'no search results'}

    ctx = "\n".join(f"[{i+1}] URL: {r['url']}\n    Title: {r.get('title','')}\n    Snippet: {r.get('snippet','')}"
                    for i, r in enumerate(results[:8]))
    prompt = f"""Given the domain "{kd}", find its OFFICIAL REGISTERED LEGAL ENTITY NAME.
Expected country: {cty}

Search results:
{ctx}

Rules:
- Return the LEGAL/REGISTERED name (e.g. "ALPHABET INC." not "Google")
- Include legal suffix (Inc., Ltd., Pvt. Ltd., PLC, etc.)
- Pick the entity that DIRECTLY OWNS this domain (not parent, not subsidiary)
- If unsure, return ""

Respond ONLY with JSON:
{{"legal_entity_name": "EXACT LEGAL NAME", "confidence": "high|medium|low", "reason": "brief"}}"""

    try:
        r = parse_json_safe(ai_call(prompt))
        found_le = r.get('legal_entity_name', '').strip()
        conf     = r.get('confidence', 'low')
        rsn      = r.get('reason', '')
        match    = fuzzy_name_match(found_le, kle) if found_le else 'NOT_FOUND'
        return {'reverse_found_le': found_le, 'reverse_match': match,
                'reverse_confidence': conf, 'reverse_reason': rsn}
    except Exception as e:
        return {'reverse_found_le': '', 'reverse_match': 'ERROR',
                'reverse_confidence': '', 'reverse_reason': str(e)}


# ---------------------------------------------------------------------------
# CHECK 3: WEBFETCH — visit the domain directly
# ---------------------------------------------------------------------------
def check_webfetch(entry: dict) -> dict:
    kd  = entry['known_domain'];  kle = entry['known_le_name']
    eid = entry['entity_id'];     cty = entry['country']

    log.info(f"  [WebFetch] {kd} ...")
    page_text = fetch_domain_pages(kd)

    # JS-only SPA — static fetch useless, mark as failed so Playwright can retry
    if page_text and 'enable javascript' in page_text.lower():
        page_text = ''

    if not page_text.strip():
        return {'webfetch_legal_name': '', 'webfetch_company_num': '',
                'webfetch_parent': '', 'webfetch_verdict': 'FETCH_FAILED',
                'webfetch_confidence': '',
                'webfetch_explanation': f'Could not fetch any pages from {kd}'}

    prompt = f"""You fetched pages from "{kd}". Verify if this domain belongs to the legal entity below.

EXPECTED ENTITY:
  Name:            {kle}
  Registration ID: {eid}
  Country:         {cty}

WEBSITE CONTENT (excerpts from /, /contact, /privacy-policy etc.):
{page_text[:7000]}

ANALYSIS TASK:
1. What LEGAL/REGISTERED company name does this site show? (check the CONTACT page, footer, privacy policy, terms, about page, company number disclosures — the registered "<Legal Co> Pvt Ltd / Inc / Ltd" + address is often only on the contact/privacy page even when the homepage shows only a brand). Treat a "<Brand> is operated by / a product of <Legal Co>" disclosure, or a matching registered name+address, as confirming the entity.
2. Any company registration number mentioned?
3. Is a PARENT COMPANY or HOLDING COMPANY mentioned?
4. Does this domain correctly belong to "{kle}"?

VERDICT — choose exactly ONE:
  "EXACT_MATCH"         : site confirms this is exactly "{kle}"
  "NAME_CHANGED"        : same legal entity but company was renamed (old name → new name)
  "PARENT_MATCH"        : domain belongs to PARENT/HOLDING company of "{kle}", OR "{kle}" is a subsidiary/holding of what the site shows — mapping still valid
  "BRAND_NAME_MISMATCH" : domain is correct but "{kle}" is brand name, not registered legal name
  "POSSIBLE_MISMATCH"   : site shows a DIFFERENT company name — relationship to "{kle}" is UNCLEAR (use this instead of MISMATCH when unsure)
  "MISMATCH"            : domain clearly belongs to a completely UNRELATED company with no corporate connection
  "NOT_FOUND"           : not enough info on site to determine

IMPORTANT: Only use "MISMATCH" when you are CERTAIN the two entities have NO corporate relationship at all.
If the site shows a different name but there COULD be a parent/subsidiary/acquisition/holding relationship, use "POSSIBLE_MISMATCH" so it can be investigated further.

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
        # If webfetch returned POSSIBLE_MISMATCH or MISMATCH, first check if
        # the names are trivially equivalent (case/abbreviation differences),
        # then try fuzzy match, and only do expensive relationship search as last resort
        if result['webfetch_verdict'] in ('POSSIBLE_MISMATCH', 'MISMATCH'):
            site_name = result.get('webfetch_legal_name', '')
            if site_name and names_are_equivalent(site_name, kle):
                result['webfetch_verdict'] = 'EXACT_MATCH'
                result['webfetch_explanation'] = (
                    f"Names match after normalization: '{site_name}' ≡ '{kle}' "
                    f"(case/abbreviation differences only)"
                )
            elif site_name and fuzzy_name_match(site_name, kle) in GOOD_VERDICTS:
                fm = fuzzy_name_match(site_name, kle)
                result['webfetch_verdict'] = fm
                result['webfetch_explanation'] = (
                    f"Names are a {fm.lower().replace('_', ' ')}: "
                    f"'{site_name}' vs '{kle}'"
                )
            else:
                result = _resolve_mismatch_via_search(entry, result)
        return result
    except Exception as e:
        return {'webfetch_legal_name': '', 'webfetch_company_num': '',
                'webfetch_parent': '', 'webfetch_verdict': 'ERROR',
                'webfetch_confidence': '', 'webfetch_explanation': str(e)}


def _resolve_mismatch_via_search(entry: dict, web_result: dict) -> dict:
    """
    Called when webfetch returns MISMATCH or POSSIBLE_MISMATCH.
    Does a targeted search to check if the relationship is actually:
      - Holding company / subsidiary
      - Acquired company (M&A)
      - Renamed entity
    Upgrades verdict to PARENT_MATCH or ACQUISITION_MATCH if confirmed,
    keeps MISMATCH only if search confirms no corporate relationship.
    """
    kle          = entry['known_le_name']
    site_entity  = web_result.get('webfetch_legal_name', '')
    kd           = entry['known_domain']

    if not site_entity:
        # Nothing to search on — keep original verdict
        return web_result

    log.info(f"  [MismatchCheck] Searching relationship: '{kle}' vs '{site_entity}' ...")

    # Search 1: Is site_entity a parent/acquirer of kle?
    q1 = f'"{kle}" acquired OR subsidiary OR "wholly owned" OR "parent company" "{site_entity}"'
    r1 = web_search(q1)

    # Search 2: Did site_entity acquire kle or is kle a holding of site_entity?
    q2 = f'"{site_entity}" acquire "{kle}" OR "{kle}" subsidiary "{site_entity}"'
    r2 = web_search(q2)

    search_snippets = ""
    for i, r in enumerate(( r1 + r2 )[:10], 1):
        search_snippets += f"[{i}] {r.get('title','')}\n    {r.get('snippet','')}\n"

    if not search_snippets.strip():
        # No results — cannot determine, keep original
        web_result['webfetch_explanation'] += " | Relationship search returned no results."
        return web_result

    prompt = f"""Two company names were found in a domain accuracy check:
  Expected LE in our data : "{kle}"
  Legal name on website   : "{site_entity}"
  Domain checked          : "{kd}"

We need to determine if these two entities are CORPORATELY RELATED
(parent/subsidiary, holding company, acquisition, rename) or completely UNRELATED.

Search results about their relationship:
{search_snippets}

Answer these questions:
1. Is "{site_entity}" the PARENT or HOLDING COMPANY of "{kle}"?
2. Did "{site_entity}" ACQUIRE "{kle}" (M&A)?
3. Is "{kle}" a SUBSIDIARY or WHOLLY-OWNED entity under "{site_entity}"?
4. Are they the SAME entity under a different name (rename)?
5. Are they COMPLETELY UNRELATED companies that just happen to have similar names?

RELATIONSHIP must be ONE of:
  "HOLDING"      : one is a holding/parent company of the other — domain mapping VALID
  "ACQUIRED"     : one acquired the other via M&A — domain mapping VALID
  "SUBSIDIARY"   : one is a subsidiary of the other — domain mapping VALID
  "RENAMED"      : same entity, different name — domain mapping VALID
  "UNRELATED"    : no corporate connection confirmed — domain mapping WRONG

Respond ONLY with JSON:
{{
  "relationship": "HOLDING|ACQUIRED|SUBSIDIARY|RENAMED|UNRELATED",
  "confidence": "high|medium|low",
  "explanation": "1-2 sentences explaining the corporate relationship found"
}}"""

    try:
        res = parse_json_safe(ai_call(prompt, max_tokens=500))
        rel  = res.get('relationship', 'UNRELATED')
        conf = res.get('confidence', 'low')
        expl = res.get('explanation', '')

        if rel in ('HOLDING', 'ACQUIRED', 'SUBSIDIARY', 'RENAMED'):
            # Upgrade verdict — mapping is actually valid
            verdict_map = {
                'HOLDING':    'PARENT_MATCH',
                'ACQUIRED':   'ACQUISITION_MATCH',
                'SUBSIDIARY': 'PARENT_MATCH',
                'RENAMED':    'NAME_CHANGED',
            }
            web_result['webfetch_verdict']     = verdict_map[rel]
            web_result['webfetch_confidence']  = conf
            web_result['webfetch_explanation'] = (
                f"Initially flagged as mismatch (site shows '{site_entity}'). "
                f"Relationship search confirmed: {expl}"
            )
            if not web_result['webfetch_parent'] and rel in ('HOLDING', 'ACQUIRED', 'SUBSIDIARY'):
                web_result['webfetch_parent'] = site_entity
        else:
            # Confirmed unrelated — keep as MISMATCH
            web_result['webfetch_verdict']     = 'MISMATCH'
            web_result['webfetch_confidence']  = conf
            web_result['webfetch_explanation'] = (
                f"Site shows '{site_entity}'. Relationship search confirms: no corporate link to '{kle}'. {expl}"
            )
    except Exception as e:
        web_result['webfetch_explanation'] += f" | Relationship search error: {e}"

    return web_result



# ---------------------------------------------------------------------------
# FINAL VERDICT — WebFetch is primary source of truth.
# Forward/Reverse are SIGNALS only, used only when WebFetch fails.
# A good WebFetch result always wins, even if fwd/rev disagree.
# ---------------------------------------------------------------------------
# Valid webfetch verdicts → mapping IS correct (domain right, entity right)
WEB_VALID   = {'EXACT_MATCH', 'NAME_CHANGED', 'PARENT_MATCH', 'ACQUISITION_MATCH', 'BRAND_NAME_MISMATCH'}
# Bad webfetch verdict → mapping IS wrong
WEB_BAD     = {'MISMATCH'}
# Inconclusive → fall back to fwd+rev signals
WEB_UNCLEAR = {'NOT_FOUND', 'FETCH_FAILED', 'ERROR', ''}

# Forward/reverse combos strong enough to trust when webfetch unclear
FWD_GOOD = {'EXACT_MATCH', 'SUBDOMAIN_MATCH'}
REV_GOOD = {'EXACT_MATCH', 'CLOSE_MATCH', 'PARTIAL_MATCH'}
FWD_BAD  = {'MISMATCH'}
REV_BAD  = {'MISMATCH'}


def compute_final_verdict(r: dict) -> dict:
    """
    Derive final_mapping_correct and final_issue_notes from all three checks.
    Priority: WebFetch > Forward+Reverse combined > NEEDS_REVIEW
    """
    web = r.get('webfetch_verdict', '').strip()
    fwd = r.get('forward_match', '').strip()
    rev = r.get('reverse_match', '').strip()
    fwd_dom  = r.get('forward_found_domain', '').strip()
    known_dom = r.get('known_domain', '').strip()
    web_legal = r.get('webfetch_legal_name', '').strip()
    web_parent = r.get('webfetch_parent', '').strip()
    site_entity = r.get('webfetch_legal_name', '').strip()
    known_le  = r.get('known_le_name', '').strip()

    notes = []

    # ── CASE 1: WebFetch gave a clear answer ────────────────────────────
    if web in WEB_VALID:
        correct = 'YES'
        if web == 'NAME_CHANGED':
            notes.append(f"Company renamed: data has old name '{known_le}', current legal name is '{web_legal}'")
        elif web in ('PARENT_MATCH', 'ACQUISITION_MATCH'):
            rel_type = 'acquired by' if web == 'ACQUISITION_MATCH' else 'parent company'
            notes.append(f"Domain valid: {rel_type} '{web_parent or site_entity}' — confirmed via relationship search")
        elif web == 'BRAND_NAME_MISMATCH':
            notes.append(f"Domain correct but LE name in data ('{known_le}') is brand name — legal name on site: '{web_legal}'")
        # Flag search-quality issues for awareness (not errors)
        if fwd not in FWD_GOOD and fwd not in ('NOT_FOUND', 'NO_RESULTS', ''):
            notes.append(f"Note: forward search found '{fwd_dom}' instead of '{known_dom}' — search quality issue, not a data error")
        if rev in REV_BAD:
            rev_le = r.get('reverse_found_le', '')
            notes.append(f"Note: reverse search found '{rev_le}' — likely old name or subsidiary, not a data error")

    elif web in WEB_BAD:
        correct = 'NO'
        site_entity = web_legal or r.get('reverse_found_le', '')
        notes.append(f"WRONG DOMAIN: site belongs to '{site_entity}', not '{known_le}'")
        if web_parent:
            notes.append(f"Site parent: {web_parent}")

    # ── CASE 2: WebFetch inconclusive — use fwd+rev signals ─────────────
    elif web in WEB_UNCLEAR:
        reason = f"WebFetch {web.lower()}"

        # Both forward and reverse agree it's correct
        if fwd in FWD_GOOD and rev in REV_GOOD:
            correct = 'YES'
            notes.append(f"{reason} — but forward+reverse both confirm mapping, treating as correct")

        # Forward correct, reverse neutral/close
        elif fwd in FWD_GOOD and rev not in REV_BAD:
            correct = 'YES'
            notes.append(f"{reason} — forward confirms, reverse inconclusive. Likely correct")

        # Forward found a DIFFERENT domain — genuine uncertainty
        elif fwd in FWD_BAD:
            correct = 'NEEDS_REVIEW'
            notes.append(f"{reason} — forward search found different domain '{fwd_dom}' instead of '{known_dom}'. Manual check needed")

        # Forward found nothing (site may block search indexing)
        elif fwd in ('NOT_FOUND', 'NO_RESULTS', ''):
            if rev in REV_GOOD:
                correct = 'YES'
                notes.append(f"{reason} — forward found nothing but reverse confirms LE. Likely correct")
            else:
                correct = 'NEEDS_REVIEW'
                notes.append(f"{reason} and forward found nothing — cannot confirm. Manual check needed")

        # Reverse says wrong LE but forward says right domain
        elif rev in REV_BAD and fwd in FWD_GOOD:
            correct = 'NEEDS_REVIEW'
            rev_le = r.get('reverse_found_le', '')
            notes.append(f"{reason} — forward correct but reverse found '{rev_le}'. May be renamed/rebranded. Manual check")

        else:
            correct = 'NEEDS_REVIEW'
            notes.append(f"{reason} — insufficient signal from forward/reverse. Manual check needed")

    else:
        correct = 'NEEDS_REVIEW'
        notes.append(f"Unexpected webfetch verdict '{web}' — manual check needed")

    return {
        'final_mapping_correct': correct,
        'final_issue_notes': '; '.join(notes) if notes else 'All checks pass',
    }

# ---------------------------------------------------------------------------
# PROCESS ONE ENTRY
# ---------------------------------------------------------------------------
def process_entry(entry: dict) -> dict:
    log.info(f"Processing [{entry['entity_id']}] {entry['known_le_name']} <-> {entry['known_domain']}")
    fwd = check_forward(entry)
    rev = check_reverse(entry)
    web = check_webfetch(entry)
    merged = {**entry, **fwd, **rev, **web}
    final = compute_final_verdict(merged)
    return {**merged, **final}


# ---------------------------------------------------------------------------
# VERDICT ICON
# ---------------------------------------------------------------------------
def vicon(v):
    if v in GOOD_VERDICTS:   return '✅'
    if v in BAD_VERDICTS:    return '❌'
    if v in EMPTY_VERDICTS:  return '⬜'
    return '⚠️'


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    global NVIDIA_API_KEYS, NVIDIA_API_KEY

    # API keys — rebuild the pool in case env was populated after import.
    if not NVIDIA_API_KEYS:
        NVIDIA_API_KEYS = _load_nvidia_keys()
        NVIDIA_API_KEY = NVIDIA_API_KEYS[0] if NVIDIA_API_KEYS else ""
    if not NVIDIA_API_KEYS:
        log.error("No NVIDIA keys! set NVIDIA_API_KEY=nvapi-... or NVIDIA_API_KEYS=nvapi-...,nvapi-...")
        sys.exit(1)
    log.info(f"NVIDIA key pool: {len(NVIDIA_API_KEYS)} key(s) loaded for rotation")

    # le.csv
    csv_path = LE_CSV
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LE_CSV)
    if not os.path.exists(csv_path):
        log.error(f"le.csv not found"); sys.exit(1)

    log.info(f"Reading {csv_path}...")
    entries = parse_le_csv(csv_path)
    log.info(f"Loaded {len(entries)} entries")

    # Output path
    out_path = OUTPUT_CSV
    if not os.path.exists(out_path):
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_CSV)
        if os.path.exists(cand):
            out_path = cand

    # ── RESUME ──────────────────────────────────────────────────────────────
    already_done = {}
    if os.path.exists(out_path):
        with open(out_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                eid = row.get('entity_id', '').strip()
                if eid:
                    already_done[eid] = dict(row)
        log.info(f"Resume: {len(already_done)} already done, skipping")
    else:
        log.info("Starting fresh")

    remaining = [e for e in entries if e['entity_id'] not in already_done]
    log.info(f"New entries to process: {len(remaining)}")

    # ── PROCESS ─────────────────────────────────────────────────────────────
    new_results = []
    if remaining:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(process_entry, e): e for e in remaining}
            try:
                for future in as_completed(futures):
                    try:
                        r = future.result()
                    except Exception as ex:
                        e = futures[future]
                        log.error(f"ERROR {e['entity_id']}: {ex}")
                        r = {**e, 'forward_found_domain': '', 'forward_match': 'ERROR',
                             'forward_confidence': '', 'forward_reason': str(ex),
                             'reverse_found_le': '', 'reverse_match': 'ERROR',
                             'reverse_confidence': '', 'reverse_reason': str(ex),
                             'webfetch_legal_name': '', 'webfetch_company_num': '',
                             'webfetch_parent': '', 'webfetch_verdict': 'ERROR',
                             'webfetch_confidence': '', 'webfetch_explanation': str(ex)}
                    new_results.append(r)
                    # Crash-safe immediate append
                    need_hdr = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
                    with open(out_path, 'a', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                        if need_hdr: w.writeheader()
                        w.writerow({k: r.get(k, '') for k in FIELDNAMES})
                    log.info(f"  ✔ [{r['entity_id']}] webfetch={r.get('webfetch_verdict','?')}")
            except KeyboardInterrupt:
                log.info("Interrupted — progress saved, safe to resume.")
                executor.shutdown(wait=False, cancel_futures=True)
    else:
        log.info("Nothing new to process.")

    # ── MERGE + REWRITE in input order ──────────────────────────────────────
    all_map = {**already_done, **{r['entity_id']: r for r in new_results}}
    id_order = {e['entity_id']: i for i, e in enumerate(entries)}
    results = sorted(
        [all_map[e['entity_id']] for e in entries if e['entity_id'] in all_map],
        key=lambda r: id_order.get(r['entity_id'], 9999)
    )
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in FIELDNAMES})
    log.info(f"Written {out_path} ({len(results)} rows)")

    # ── APPLY final verdict to any already_done rows loaded from CSV ───────
    # (new rows already have it from process_entry; old rows from CSV need it computed)
    for r in results:
        if not r.get('final_mapping_correct'):
            fv = compute_final_verdict(r)
            r.update(fv)

    # Rewrite with final columns populated
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in FIELDNAMES})

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    total = len(results)
    if not total: return

    def cnt(field, vals):
        return sum(1 for r in results if r.get(field) in vals)

    yes_count    = cnt('final_mapping_correct', {'YES'})
    no_count     = cnt('final_mapping_correct', {'NO'})
    review_count = cnt('final_mapping_correct', {'NEEDS_REVIEW'})

    # WebFetch breakdown (for diagnostics)
    w_exact  = cnt('webfetch_verdict',{'EXACT_MATCH'})
    w_rename = cnt('webfetch_verdict',{'NAME_CHANGED'})
    w_parent = cnt('webfetch_verdict',{'PARENT_MATCH'})
    w_brand  = cnt('webfetch_verdict',{'BRAND_NAME_MISMATCH'})
    w_bad    = cnt('webfetch_verdict',{'MISMATCH'})
    w_miss   = cnt('webfetch_verdict',{'NOT_FOUND','FETCH_FAILED','ERROR',''})

    print("\n" + "═"*70)
    print(f"  FINAL ACCURACY REPORT  ({total} entities)")
    print("═"*70)
    print(f"  ✅ Mapping CORRECT      : {yes_count}/{total} ({yes_count/total*100:.1f}%)")
    print(f"  ❌ Mapping WRONG        : {no_count}/{total} ({no_count/total*100:.1f}%)")
    print(f"  ⚠️  Needs manual review  : {review_count}/{total} ({review_count/total*100:.1f}%)")

    print(f"\n  WebFetch breakdown (diagnostic):")
    print(f"    Exact match      : {w_exact}")
    print(f"    Renamed (valid)  : {w_rename}")
    print(f"    Parent (valid)   : {w_parent}")
    print(f"    Brand name       : {w_brand}")
    print(f"    Mismatch (wrong) : {w_bad}")
    print(f"    Fetch failed     : {w_miss}")

    # ── Per-entity table ─────────────────────────────────────────────────
    def ficon(v):
        return {'YES': '✅', 'NO': '❌', 'NEEDS_REVIEW': '⚠️ '}.get(v, '❓')

    print("\n" + "─"*90)
    print(f"  {'Entity ID':<26} {'LE Name':<35} {'FINAL':<6} {'Issue / Notes'}")
    print("─"*90)

    # Print in order: NO first, then NEEDS_REVIEW, then YES
    for status in ['NO', 'NEEDS_REVIEW', 'YES']:
        for r in results:
            if r.get('final_mapping_correct') != status:
                continue
            icon  = ficon(status)
            eid   = r['entity_id'][:25]
            le    = r['known_le_name'][:34]
            notes = r.get('final_issue_notes','')[:80]
            print(f"  {eid:<26} {le:<35} {icon:<6} {notes}")

    print("═"*90)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
