#!/usr/bin/env python
"""
Varsha US LE map — Domain → LE + accuracy check (CLI, parallel, resume-safe).

Input  : varsha us revenue le map .xlsx  (col: Website)
Output : varsha_us_le_results.xlsx
Status : printed every row, flushed to disk every 20 rows.
"""
import os, sys, types, time, threading, re
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "NVIDIA_API_KEY",
    "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",
)

# --- Mock streamlit so we can import app.py outside a streamlit runtime -----
class _NoOp:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _NoOp()
    def __getattr__(self, name): return _NoOp()
    def __getitem__(self, k): return _NoOp()
    def __setitem__(self, k, v): pass
    def __contains__(self, k): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __bool__(self): return False

_st = types.ModuleType("streamlit")
_st.session_state = {}
_st.secrets = {}
_st.cache_data = lambda *a, **kw: (lambda f: f)
_st.cache_resource = lambda *a, **kw: (lambda f: f)
for attr in ("set_page_config","title","subheader","header","write","markdown","text",
             "radio","selectbox","checkbox","button","text_input","number_input",
             "file_uploader","progress","empty","dataframe","table","info","error",
             "success","warning","metric","expander","code","columns","download_button",
             "sidebar","container","tabs","stop","spinner","status"):
    setattr(_st, attr, _NoOp())
sys.modules["streamlit"] = _st

# --- Now safe to import the app's functions ---------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import find_le_from_domain, run_accuracy_check_single  # noqa: E402

import pandas as pd  # noqa: E402

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
XLSX_IN     = os.path.join(os.path.dirname(BASE_DIR), "varsha us revenue le map .xlsx")
XLSX_OUT    = os.path.join(os.path.dirname(BASE_DIR), "varsha_us_le_results.xlsx")
WORKERS     = int(os.environ.get("WORKERS", "6"))
SAVE_EVERY  = 20
DEFAULT_COUNTRY = "United States"

HEADERS = [
    "website","domain","country",
    "le_name","cin","confidence","reason",
    "fwd_le","rev_le","web_le",
    "forward_found_domain","forward_match","forward_confidence",
    "reverse_found_le","reverse_match","reverse_confidence",
    "webfetch_legal_name","webfetch_company_num","webfetch_verdict",
    "webfetch_confidence","webfetch_explanation",
    "final_mapping_correct","final_issue_notes",
]


def _clean_domain(d: str) -> str:
    d = (d or "").strip().lower()
    if not d or d == "nan":
        return ""
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    return d.split("/")[0].split("?")[0].split("#")[0]


def _process(entry):
    t0 = time.time()
    domain = entry["domain"]
    fl = find_le_from_domain(domain, entry["country"])
    le_name = fl.get("le_name", "")

    acc = {}
    if le_name:
        acc = run_accuracy_check_single(
            domain, le_name,
            fl.get("country", entry["country"]),
            fl.get("cin", ""),
            do_ch=False,
        )

    out = {
        "website": entry["website"],
        "domain":  domain,
        "country": entry["country"],
        "le_name": le_name,
        "cin":     fl.get("cin", ""),
        "confidence": fl.get("confidence", ""),
        "reason":     fl.get("reason", ""),
        "fwd_le": fl.get("fwd_le", ""),
        "rev_le": fl.get("rev_le", ""),
        "web_le": fl.get("web_le", ""),
        "forward_found_domain":  acc.get("forward_found_domain", ""),
        "forward_match":         acc.get("forward_match", ""),
        "forward_confidence":    acc.get("forward_confidence", ""),
        "reverse_found_le":      acc.get("reverse_found_le", ""),
        "reverse_match":         acc.get("reverse_match", ""),
        "reverse_confidence":    acc.get("reverse_confidence", ""),
        "webfetch_legal_name":   acc.get("webfetch_legal_name", ""),
        "webfetch_company_num":  acc.get("webfetch_company_num", ""),
        "webfetch_verdict":      acc.get("webfetch_verdict", ""),
        "webfetch_confidence":   acc.get("webfetch_confidence", ""),
        "webfetch_explanation":  acc.get("webfetch_explanation", ""),
        "final_mapping_correct": acc.get("final_mapping_correct", ""),
        "final_issue_notes":     acc.get("final_issue_notes", ""),
        "_elapsed": f"{time.time()-t0:.1f}s",
    }
    return out


def main():
    df_in = pd.read_excel(XLSX_IN)
    df_in.columns = [c.strip() for c in df_in.columns]
    print(f"loaded {len(df_in)} rows from {XLSX_IN}", flush=True)
    print(f"cols: {list(df_in.columns)}", flush=True)

    done = set()
    prev = []
    if os.path.exists(XLSX_OUT):
        try:
            df_done = pd.read_excel(XLSX_OUT)
            dom = df_done.get("domain", "").astype(str).str.strip().str.lower()
            keep = df_done[(dom != "") & (dom != "nan")]
            done = set(keep["domain"].astype(str).str.strip().str.lower())
            prev = keep.to_dict("records")
            print(f"resuming — {len(done)} domains already done", flush=True)
        except Exception as e:
            print(f"could not load previous results: {e}", flush=True)

    entries = []
    for _, row in df_in.iterrows():
        website = str(row.get("Website", "")).strip()
        domain = _clean_domain(website)
        if not domain:
            continue
        if domain in done:
            continue
        entries.append({"website": website, "domain": domain, "country": DEFAULT_COUNTRY})

    print(f"to process: {len(entries)} | workers={WORKERS} | country={DEFAULT_COUNTRY}", flush=True)

    results = list(prev)
    lock = threading.Lock()
    completed = [0]
    found = [0]
    yes = [0]

    def _flush():
        df_out = pd.DataFrame(results)
        for h in HEADERS:
            if h not in df_out.columns:
                df_out[h] = ""
        df_out[HEADERS].to_excel(XLSX_OUT, index=False)

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(_process, e): e for e in entries}
            for fut in as_completed(futures):
                try:
                    out = fut.result()
                except Exception as err:
                    e = futures[fut]
                    out = {"website": e["website"], "domain": e["domain"],
                           "country": e["country"], "le_name": "",
                           "final_mapping_correct": "ERROR",
                           "final_issue_notes": str(err)}
                with lock:
                    results.append(out)
                    completed[0] += 1
                    if out.get("le_name"): found[0] += 1
                    if out.get("final_mapping_correct") == "YES": yes[0] += 1
                    print(
                        f"[{completed[0]}/{len(entries)}] "
                        f"{out['domain'][:35]:35s} → "
                        f"{(out.get('le_name') or '—')[:40]:40s} "
                        f"verdict={(out.get('final_mapping_correct') or '—'):14s} "
                        f"[{out.get('_elapsed','')}] "
                        f"found={found[0]} yes={yes[0]}",
                        flush=True,
                    )
                    if completed[0] % SAVE_EVERY == 0:
                        _flush()
                        print(f"  → saved {len(results)} rows → {XLSX_OUT}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — flushing partial …", flush=True)

    _flush()
    print(f"\nDone! LE found {found[0]}/{len(entries)} | YES verdict {yes[0]}", flush=True)
    print(f"output: {XLSX_OUT}", flush=True)


if __name__ == "__main__":
    main()
