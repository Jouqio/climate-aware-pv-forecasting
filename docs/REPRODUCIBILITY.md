# Reproducibility

## Environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Random Seeds

| Component | Seed | Location |
|---|---|---|
| Stochastic target reconstruction (Y_stoch, 7 loss components) | `42` | `notebooks/02_target_reconstruction.py` |
| XGBoost model fitting | `42` (primary), `0..199` (bootstrap PI resamples) | `notebooks/07_xgboost_model.py` |

No other component (OLS, SARIMAX) involves randomness.

**Known non-determinism caveat:** despite the fixed `seed=42`, minor
numeric variation (typically in the 3rd–4th decimal place) has been
observed across independent re-runs, most likely from BLAS/threading
non-determinism during XGBoost's fold-1 hyperparameter search. This
does not change any qualitative conclusion (leakage direction/range,
meta-analysis significance, KW non-significance) but means exact
decimal reproduction across different hardware/library versions is not
guaranteed. Always regenerate `results/combined/` from a fresh pipeline
run rather than assuming numbers in the manuscript text are
byte-identical to a specific historical run.

## Exact Replication Steps

```bash
git clone https://github.com/Jouqio/climate-aware-pv-forecasting.git
cd climate-aware-pv-forecasting
pip install -r requirements.txt
python run_all_locations.py
```

This reproduces every number in `results/<site>/outputs/*.csv` and
`results/combined/*` from the tracked `data/raw/*.csv` files — no
external data download or API key required.

## Verifying No Leakage Regression

Every notebook calls `utils.verify_no_feature_leakage()` at the start
of every walk-forward fold. This asserts that per-fold training-window
climatology differs numerically from full-sample climatology. A failed
assertion means a future code change has reintroduced the leakage this
architecture was built to prevent — treat it as release-blocking.

## Cross-Checking Manuscript Numbers

Before citing any number from
`docs/manuscript_10sites/MANUSCRIPT_FULL_10SITES.md`, verify it against
the corresponding `results/combined/TABLE*.csv` — the manuscript text
is generated FROM these tables, not the other way around. If they ever
disagree, the CSV (freshest pipeline run) is authoritative.

## Known Sources of Numeric Non-Determinism

- **XGBoost hyperparameter search** (`notebooks/07_xgboost_model.py`,
  Part A) uses an inner train/validation split confined to fold-1's
  training window (2005–2012 / 2013–2014). Re-running on different
  hardware/library versions may select marginally different
  `best_params`; the exact values used are saved to
  `results/<site>/outputs/07_xgboost_best_params.json`.
- **SARIMAX convergence** can occasionally fail for a specific fold;
  the walk-forward loop logs and skips such folds rather than
  crashing (check the `converged` column in
  `results/<site>/outputs/06_sarimax_walkforward_results.csv`).

## Verifying Cross-Site Consistency

After running all sites, `results/combined/statistical_summary.csv`
reports the meta-analysis and between-site test results. If you add a
new site and a previously-consistent finding changes materially,
first re-verify `data/raw/<newsite>.csv`'s format against
`docs/DATA_STRUCTURE.md` before treating it as a genuine new finding.
