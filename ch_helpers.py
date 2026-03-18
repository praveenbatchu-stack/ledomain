"""
Companies House API helpers — extracted for Streamlit Cloud deployment.
"""

import os
import re
import time
import random
import threading
import requests
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# CONFIG — keys loaded from env / st.secrets at runtime
# ---------------------------------------------------------------------------
CH_API_KEYS = []
CH_BASE = "https://api.company-information.service.gov.uk"

_ch_lock = threading.Lock()
_ch_last_call = [0.0]
CH_MIN_GAP = 0.15


def init_ch_keys(keys=None):
    """Initialize CH API keys from a list or from environment."""
    global CH_API_KEYS
    if keys:
        CH_API_KEYS = list(keys)
    elif not CH_API_KEYS:
        env_keys = os.environ.get("CH_API_KEYS", "")
        if env_keys:
            CH_API_KEYS = [k.strip() for k in env_keys.split(",") if k.strip()]


def ch_get(path, retries=3):
    if not CH_API_KEYS:
        return None
    with _ch_lock:
        now = time.time()
        gap = CH_MIN_GAP - (now - _ch_last_call[0])
        if gap > 0:
            time.sleep(gap)
        _ch_last_call[0] = time.time()

    for attempt in range(retries):
        key = random.choice(CH_API_KEYS)
        try:
            resp = requests.get(f"{CH_BASE}{path}", auth=(key, ''), timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            if resp.status_code == 404:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return None


def ch_search(query, limit=10):
    data = ch_get(f"/search/companies?q={quote_plus(query)}&items_per_page={limit}")
    if not data:
        return []
    return data.get('items', [])


def _suffix(name):
    name = name.strip()
    for suf in ['LIMITED', 'LTD', 'PLC', 'LLP', 'LP', 'CIC', 'CIO']:
        if name.upper().endswith(suf):
            return suf
    return ''


def _names_match_exact(a, b):
    a = re.sub(r'[.\s]+$', '', a.strip().upper())
    b = re.sub(r'[.\s]+$', '', b.strip().upper())
    return a == b


def _strip_spaces(name):
    return re.sub(r'\s+', '', name.upper())


def _base_no_spaces(name):
    return _strip_spaces(name.upper().replace(_suffix(name), '').strip())


def ch_verify_exact(le_name, ai_cin=''):
    """Verify LE name against Companies House with exact matching."""
    if not CH_API_KEYS:
        return '', '', 'no_keys'

    if ai_cin and re.match(r'^\d{6,8}$|^[A-Z]{2}\d{6}$', ai_cin.strip()):
        profile = ch_get(f"/company/{ai_cin.strip()}")
        if profile:
            ch_name = profile.get('company_name', '')
            if _names_match_exact(le_name, ch_name):
                return ai_cin.strip(), ch_name, 'verified'
            if _base_no_spaces(le_name) == _base_no_spaces(ch_name):
                return ai_cin.strip(), ch_name, 'verified'

    # Search CH
    items = ch_search(le_name)
    if not items:
        suf = _suffix(le_name)
        base = le_name[:-(len(suf))].strip() if suf else le_name
        alt_suffix = {'LTD': 'LIMITED', 'LIMITED': 'LTD'}.get(suf.upper(), '')
        if alt_suffix:
            items = ch_search(f"{base} {alt_suffix}")
        if not items and base:
            items = ch_search(base)

    if not items:
        return '', '', 'not_found'

    target_base_nospace = _base_no_spaces(le_name)

    # Pass 1: Exact match
    for item in items:
        ch_name = item.get('title', '')
        if _names_match_exact(le_name, ch_name):
            return item.get('company_number', ''), ch_name, 'verified'

    # Pass 2: Space-normalized + same suffix
    target_suffix = _suffix(le_name)
    for item in items:
        ch_name = item.get('title', '')
        ch_suffix = _suffix(ch_name)
        if _base_no_spaces(ch_name) == target_base_nospace and ch_suffix.upper() == target_suffix.upper():
            return item.get('company_number', ''), ch_name, 'verified'

    # Pass 3: Space-normalized + active
    for item in items:
        ch_name = item.get('title', '')
        if _base_no_spaces(ch_name) == target_base_nospace and item.get('company_status') == 'active':
            return item.get('company_number', ''), ch_name, 'partial'

    return '', '', 'not_found'
