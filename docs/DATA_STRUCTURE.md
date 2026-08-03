# Data Structure

## Standardised Raw Input: `data/raw/<city>.csv`

Every study site's raw monthly meteorological data lives at
`data/raw/<city>.csv`, where `<city>` is the **lowercase** site name
used as the `PV_LOCATION` value throughout the pipeline.

Currently tracked (10 sites):
```
data/raw/medan.csv      data/raw/pekanbaru.csv   data/raw/pontianak.csv
data/raw/bontang.csv    data/raw/samarinda.csv   data/raw/balikpapan.csv
data/raw/makassar.csv   data/raw/surabaya.csv    data/raw/kupang.csv
data/raw/jayapura.csv
```

### Exact NASA POWER Export Settings

Download from <https://power.larc.nasa.gov/data-access-viewer/> with:

| Setting | Value |
|---|---|
| Temporal resolution | Monthly & Annual |
| User Community | Renewable Energy (RE) |
| Time Standard | UTC |
| Date range | 2005-01-01 to 2025-12-31 |
| Wind Elevation | 10 m |
| Wind Surface | Savanna (18-m broadleaf trees, 30% groundcover) |
| Pressure Correction | 0 hPa |
| Output format | CSV |

**Required (core) parameters** — present at every site, pipeline asserts
this and fails with a clear error otherwise:
```
ALLSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DNI, ALLSKY_SFC_SW_DIFF, ALLSKY_KT,
CLOUD_AMT, IMERG_PRECTOT, RH2M, T2M, TS, WS10M, WSC, PS
```

**Optional parameters** — vary by site, handled automatically:
```
PSC          — present at Bontang only
T2M_MAX, T2M_MIN  — present at the other 9 sites
```
Neither optional parameter is used in the modelling feature set.

### File Format

NASA POWER's CSV export is "wide" format: one row per parameter per
year, one column per calendar month (JAN…DEC), preceded by a multi-line
text header (`-BEGIN HEADER-` … `-END HEADER-`).
`notebooks/01_data_preprocessing.py` parses this directly.

### Adding a New Site (Auto-Discovery)

1. Download a CSV per the settings above.
2. Save as `data/raw/<newcity>.csv`.
3. Run `python run_all_locations.py --sites newcity` — **no code changes**,
   the pipeline auto-discovers every CSV in `data/raw/`.
4. To include it in the default full-run and `_CANONICAL_ORDER` display
   order, add its name to `_CANONICAL_ORDER` in `run_all_locations.py`
   and `SITE_META` in `scripts/cross_site_comparison.py` (province +
   climate regime label, used for figure colouring only).

## Per-Site Results: `results/<site>/`

```
results/<site>/
├── data/        intermediate parquet (gitignored — regenerate via notebooks 01-03)
├── outputs/     numeric results as CSV (tracked)
└── figures/     13 per-site figures as PNG (tracked)
```

## Combined Results: `results/combined/`

Produced by `scripts/cross_site_comparison.py`, reading every available
site's `results/<site>/outputs/*.csv`:
```
results/combined/
├── TABLE1_dataset_summary.csv
├── TABLE2_leakage_all_sites.csv
├── TABLE3_ols_meta_analysis.csv
├── TABLE4_walkforward_wilcoxon.csv
├── TABLE5_shap_rankings.csv
├── TABLE6_statistical_significance.csv
├── TABLE7_climate_statistics.csv
├── ALL_TABLES_publication_ready.xlsx   (all 7 tables, one workbook)
├── statistical_summary.csv             (meta-analysis + KW test raw values)
├── cross_site_leakage.csv / ols_coefficients_all_sites.csv / ...
└── figures/     (12 cross-site comparison figures)
```
Entirely regeneratable — never edit files under `results/combined/` by
hand; re-run `python scripts/cross_site_comparison.py` instead.
