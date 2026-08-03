# Experiment Workflow

Each of the 10 notebooks is a standalone Python script
(`python notebooks/NN_name.py`), runnable individually or orchestrated
via `run_all_locations.py`. Every notebook is **location-agnostic**:
behaviour depends only on the `PV_LOCATION` environment variable.

## Pipeline Steps

| # | Notebook | Purpose |
|---|---|---|
| 01 | `data_preprocessing` | Parse `data/raw/<site>.csv` → clean monthly panel; QC checks |
| 02 | `target_reconstruction` | Build Y_det (leakage proof) and Y_stoch (7-component stochastic target) |
| 03 | `feature_engineering` | 12-feature set; anomaly features deferred to per-fold computation (leakage-free) |
| 04 | `validation_framework` | Define 9 walk-forward folds; feature-level leakage guard verification |
| 05 | `ols_hc3_model` | OLS-HC3: full 12-feature (diagnostic) + low-VIF 8-feature (primary) specs |
| 06 | `sarimax_climate_model` | SARIMAX(0,0,1)(0,0,0)₁₂ + ONI exogenous; prediction intervals |
| 07 | `xgboost_model` | XGBoost; fold-1-only hyperparameter search; bootstrap PI |
| 08 | `shap_analysis` | SHAP TreeExplainer; cross-fold stability check |
| 09 | `residual_diagnostics` | Diebold-Mariano, Friedman, Wilcoxon tests vs. climatology |
| 10 | `figure_generation` | 13 per-site publication figures |

After all sites complete steps 01–09:

| Script | Purpose |
|---|---|
| `scripts/cross_site_comparison.py` | Auto-discovers every processed site → `results/combined/`: meta-analysis, leakage comparison, 7 tables, 12 figures |

## Running the Workflow

```bash
# Everything, all discovered sites, in order:
python run_all_locations.py

# One site, full pipeline:
python run_all_locations.py --sites bontang

# One site, cheap steps only (skip XGBoost/SHAP):
python run_all_locations.py --sites bontang --steps 1-5

# Manually, one notebook at a time:
export PV_LOCATION=bontang
python notebooks/01_data_preprocessing.py
python notebooks/02_target_reconstruction.py
# ... etc.
```

## Design Principle: Per-Fold, Not Full-Sample, Anomalies

Steps 03–09 compute climatological anomaly features (GHI_anom,
CLOUD_anom, PRECTOT_anom) **inside each walk-forward fold**, using only
that fold's training window. This is the pipeline's central
leakage-prevention mechanism — an earlier version computed these
features once on the full dataset, silently leaking future information
into every fold. `utils.verify_no_feature_leakage()` runs at the start
of every fold, every notebook, every site, and raises an assertion
error if this discipline is ever broken.

## Design Principle: Why Ten Sites?

The pipeline was originally validated at one site (Bontang) then
expanded to 10 sites across 6 climate regimes specifically to test
whether leakage magnitude, coefficient stability, and forecastability
patterns are universal or climate-dependent (see
`docs/manuscript_10sites/MANUSCRIPT_FULL_10SITES.md` Section 4 for the
answer: leakage is universal but its *magnitude* is climate-dependent;
GHI_anom's coefficient is quantitatively stable; forecastability and
SHAP attribution are majority-consistent but not universal).
