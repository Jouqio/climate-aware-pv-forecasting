#!/usr/bin/env python3
"""Multi-site pipeline orchestrator. Auto-discovers sites from data/raw/*.csv."""
import argparse, subprocess, sys, time
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DATA_RAW_DIR  = REPO_ROOT / "data" / "raw"

_CANONICAL_ORDER = [
    "medan","pekanbaru","pontianak","bontang","samarinda",
    "balikpapan","makassar","surabaya","kupang","jayapura",
]

def _discover_sites():
    if not DATA_RAW_DIR.exists():
        return _CANONICAL_ORDER
    found = {p.stem for p in DATA_RAW_DIR.glob("*.csv")}
    ordered = [s for s in _CANONICAL_ORDER if s in found]
    ordered += sorted(found - set(_CANONICAL_ORDER))
    return ordered

DEFAULT_SITES = _discover_sites()

ALL_NOTEBOOKS = [
    "01_data_preprocessing","02_target_reconstruction",
    "03_feature_engineering","04_validation_framework",
    "05_ols_hc3_model","06_sarimax_climate_model",
    "07_xgboost_model","08_shap_analysis",
    "09_residual_diagnostics","10_figure_generation",
]

def parse_steps(spec):
    if spec is None: return ALL_NOTEBOOKS
    nums = set()
    for part in spec.split(","):
        if "-" in part:
            lo,hi = part.split("-"); nums.update(range(int(lo),int(hi)+1))
        else:
            nums.add(int(part))
    return [nb for nb in ALL_NOTEBOOKS if int(nb[:2]) in nums]

def run_nb(nb, site):
    import os
    script = NOTEBOOKS_DIR / f"{nb}.py"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ,"PV_LOCATION":site},
        capture_output=True, text=True
    )
    elapsed = time.time()-t0
    if proc.returncode != 0:
        print(f"    ❌ {nb} FAILED ({elapsed:.0f}s)")
        for line in (proc.stderr+proc.stdout).strip().split("\n")[-20:]:
            print(f"    | {line}")
        return False, elapsed
    print(f"    ✅ {nb} ({elapsed:.0f}s)")
    return True, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--steps", default=None)
    parser.add_argument("--no-combine", action="store_true")
    args = parser.parse_args()

    notebooks = parse_steps(args.steps)
    sites     = [s.strip().lower() for s in args.sites]
    print(f"Sites ({len(sites)}): {sites}")
    print(f"Steps ({len(notebooks)}): {[nb[:2] for nb in notebooks]}")

    summaries = []
    t_total = time.time()
    for site in sites:
        raw_csv = DATA_RAW_DIR / f"{site}.csv"
        print(f"\n{'═'*60}\n  {site}\n{'═'*60}")
        if not raw_csv.exists():
            avail = [p.stem for p in DATA_RAW_DIR.glob("*.csv")]
            print(f"  ⚠ SKIP: {raw_csv} not found. Available: {avail}")
            summaries.append((site,"skipped",0.0)); continue
        t0=time.time(); ok=True
        for nb in notebooks:
            success,_ = run_nb(nb, site)
            if not success: ok=False; break
        elapsed=time.time()-t0
        status="success" if ok else "failed"
        summaries.append((site,status,elapsed))

    print(f"\n{'═'*60}\n  SUMMARY\n{'═'*60}")
    for site,status,t in summaries:
        icon={"success":"✅","failed":"❌","skipped":"⚠️"}[status]
        print(f"  {icon} {site:<14} {status}  [{t:.0f}s]")

    n_ok = sum(1 for _,s,_ in summaries if s=="success")
    print(f"\n  {n_ok}/{len(summaries)} completed. Total: {time.time()-t_total:.0f}s")

    if not args.no_combine and notebooks==ALL_NOTEBOOKS and n_ok>=1:
        print("\nRunning cross-site comparison...")
        subprocess.run([sys.executable, str(REPO_ROOT/"scripts"/"cross_site_comparison.py")])

    if any(s=="failed" for _,s,_ in summaries):
        sys.exit(1)

if __name__=="__main__":
    main()
