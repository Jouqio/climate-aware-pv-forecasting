# Multi-Site Execution

## The `PV_LOCATION` Mechanism

Every notebook resolves its data and output paths based on a single
environment variable, `PV_LOCATION`. This is the entire mechanism that
makes the pipeline multi-site:

```python
# Standard path-resolution block in every notebook:
_PV_LOCATION = os.environ.get("PV_LOCATION", "").strip().lower()
_REPO_ROOT = Path(__file__).resolve().parent.parent

if _PV_LOCATION:
    BASE_DIR = _REPO_ROOT / "results" / _PV_LOCATION   # e.g. results/bontang/
else:
    BASE_DIR = Path(__file__).resolve().parent          # legacy: notebooks/
```

Input for notebook 01 similarly resolves to `data/raw/<PV_LOCATION>.csv`
when set. Setting `PV_LOCATION` to a different value re-runs the
**identical** code against a different site's data, writing to a
different `results/` subfolder — no cross-contamination is possible
because the paths are fully isolated per location.

## Auto-Discovery

`run_all_locations.py` and `scripts/cross_site_comparison.py` both
auto-discover sites from `data/raw/*.csv` — no hardcoded site list to
edit when adding a new location:

```python
def _discover_sites():
    found = {p.stem for p in DATA_RAW_DIR.glob("*.csv")}
    ordered = [s for s in _CANONICAL_ORDER if s in found]   # known sites, W-E order
    ordered += sorted(found - set(_CANONICAL_ORDER))        # any new/unknown sites
    return ordered
```

## Running Multiple Sites

```bash
python run_all_locations.py                          # all discovered sites
python run_all_locations.py --sites bontang makassar  # subset
python run_all_locations.py --steps 1-5                # partial pipeline
python run_all_locations.py --no-combine                # skip cross-site step
```

This orchestrator:
1. Validates `data/raw/<site>.csv` exists before attempting each site
   (skips with a clear warning listing available sites if missing).
2. Runs notebooks 01→10 (or `--steps` subset) as subprocesses, one site
   at a time, with `PV_LOCATION` set in each subprocess's environment.
3. Prints a ✅/❌/⚠️ summary per site.
4. Automatically runs `scripts/cross_site_comparison.py` if at least
   one site completed the full pipeline.
5. Exits non-zero if any site failed (for CI integration).

### Manual / debugging

```bash
export PV_LOCATION=samarinda
for nb in notebooks/0*.py notebooks/1*.py; do
    python "$nb" || { echo "FAILED at $nb"; break; }
done
```

### Parallelising

Each site's run is fully independent (isolated paths, isolated
subprocess):
```bash
for site in bontang samarinda balikpapan pontianak makassar; do
    PV_LOCATION=$site python run_pipeline.py &
done
wait
python scripts/cross_site_comparison.py
```

## Current 10 Sites (2005–2025 NASA POWER, verified)

```
medan  pekanbaru  pontianak  bontang  samarinda
balikpapan  makassar  surabaya  kupang  jayapura
```

## Adding an 11th Site

No code changes required:
1. `data/raw/newcity.csv`
2. `python run_all_locations.py --sites newcity`
3. `python scripts/cross_site_comparison.py` (to include it in combined outputs)
