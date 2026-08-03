# Migration Guide

This document explains how the repository evolved and how to update
any external scripts that depend on an older layout.

## Architecture History

| Phase | Layout | Sites | Status |
|---|---|---|---|
| 1. Original | `notebooks/{data,outputs,figures}/`, Bontang hardcoded | 1 (Bontang) | Legacy, still works |
| 2. Q1 leakage audit | Same layout; `utils.py` shared metrics module; critical leakage bugs fixed | 1 | Superseded |
| 3. Multi-location v1 | `notebooks/locations/<site>/...`; `PV_LOCATION` introduced | 5 | Superseded |
| 4. Standardised multi-site | `data/raw/<city>.csv` + `results/<site>/...` | 5 | Superseded |
| 5. **Current: 10-site auto-discovery** | Same structure as Phase 4, auto-discovery added | **10** | **Current** |

## What Changed Phase 4 → 5

- `run_all_locations.py`: `DEFAULT_SITES` hardcoded list replaced by
  `_discover_sites()`, which scans `data/raw/*.csv` automatically.
- `scripts/cross_site_comparison.py`: same auto-discovery; `SITE_META`
  extended to 10 sites / 6 climate regimes.
- 5 new NASA POWER CSVs added: Medan, Pekanbaru, Surabaya, Kupang,
  Jayapura (alongside the existing Bontang, Samarinda, Balikpapan,
  Pontianak, Makassar).
- `.gitignore` bug fixed: the rule `figures/` (no leading slash)
  was matching every directory named `figures` anywhere in the repo,
  silently excluding `results/<site>/figures/` and
  `results/combined/figures/` publication outputs. Changed to
  `/notebooks/figures/` (repo-root-anchored) to target only the
  intended legacy directory.
- Repository root cleaned: 11 legacy single-site-era working documents
  moved to `docs/archive/single_site_era/` (see that folder's README
  for details and provenance).

## Backward Compatibility

Every notebook still works with `PV_LOCATION` unset, falling back to
the original single-site layout (`notebooks/data/`, `notebooks/outputs/`,
`notebooks/figures/`). `run_pipeline.py` (the original orchestrator)
remains fully functional in this mode.

```bash
# Legacy style, still works:
python run_pipeline.py
```

## Migrating Custom Scripts

| Old path (any phase 1-4) | Current path |
|---|---|
| `notebooks/data/nasa_power_monthly_bontang_2005_2025.csv` | `data/raw/bontang.csv` |
| `notebooks/locations/<site>/outputs/` | `results/<site>/outputs/` |
| `notebooks/locations/<site>/figures/` | `results/<site>/figures/` |
| (no cross-site tooling existed) | `results/combined/` |

## Verifying This Migration Introduced No Regression

The core empirical result (GHI_anom coefficient at Bontang,
β ≈ +0.087, p ≈ 0.0017) has been independently re-verified after every
architectural change across all 5 phases. If auditing this migration,
compare `results/bontang/outputs/05_ols_coefficients.csv` against the
value quoted in `docs/manuscript_10sites/MANUSCRIPT_FULL_10SITES.md`
Table 3.
