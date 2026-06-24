"""
=============================================================================
NOTEBOOK 07: XGBOOST MODEL
=============================================================================
Purpose  : Constrained XGBoost for monthly PV forecasting.
           Deliberately conservative hyperparameters for n=240 training obs..
           Walk-forward evaluation with IDENTICAL splits as OLS/SARIMAX.
           Bootstrap prediction intervals.

CONSTRAINT RATIONALE (n=240 training, 12 features):
  max_depth ≤ 5     : prevents memorizing 12-obs seasonal patterns
  n_estimators ≤ 200: with small data, more trees = more overfitting
  subsample = 0.8   : row subsampling prevents overfitting
  colsample_bytree = 0.8 : feature subsampling
  learning_rate = 0.05: slow learning → more robust

Input    : data/03_model_ready.parquet
           data/03_final_features.csv
Output   : outputs/07_xgboost_walkforward_results.csv
           outputs/07_xgboost_feature_importance.csv
           data/07_xgboost_predictions.parquet

=============================================================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR.parent))  # utils.py lives at repo root, one level above notebooks/
from utils import get_split_data, skill_score, verify_no_feature_leakage  # noqa: E402

df = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
FINAL_FEATURES = pd.read_csv(f"{DATA_DIR}/03_final_features.csv")["feature"].tolist()
TARGET = "Y_stoch"
RAW_ANOMALY_SOURCE_COLS = ["GHI", "CLOUD", "PRECTOT", "T2M"]

print(f"Loaded: {len(df)} observations, {len(FINAL_FEATURES)} features")

# ══════════════════════════════════════════════════════════════════════════
# PART A: HYPERPARAMETER SEARCH (on first training fold only)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: HYPERPARAMETER SEARCH (fold 1 only, 2005-2014 train)")
print("=" * 60)

"""
IMPORTANT: hyperparameter search ONLY on fold 1 data.
Using all folds for tuning would constitute temporal leakage.
We use the first 10-year training window as our tuning set.
Validation within this window: last 2 years (2013-2014) as inner holdout.
"""

# Fold 1: train 2005-2012 inner train, 2013-2014 inner val
inner_train = df[df["YEAR"] <= 2012].copy()
inner_val   = df[(df["YEAR"] >= 2013) & (df["YEAR"] <= 2014)].copy()

# Q1 AUDIT FIX: recompute anomaly-derived features for the hyperparameter
# search using ONLY inner_train's climatology (2005-2012), applied to
# both inner_train and inner_val. The original code used the *_anom
# columns precomputed once on the full 2005-2025 sample in notebook 03 —
# this would leak 2013-2025 information into a tuning step that is
# supposed to see only 2005-2012. This fix extends the same per-fold
# discipline used in the main walk-forward loop (Part B) down to the
# hyperparameter search itself.
from utils import expanding_climatology
_clim = expanding_climatology(inner_train, inner_val, RAW_ANOMALY_SOURCE_COLS)
for col in RAW_ANOMALY_SOURCE_COLS:
    inner_train[f"{col}_anom"] = _clim["train_anom"][f"{col}_anom"].values
    inner_val[f"{col}_anom"]   = _clim["test_anom"][f"{col}_anom"].values
inner_train["ONI_x_CLOUD_anom"] = inner_train["ONI"] * inner_train["CLOUD_anom"]
inner_val["ONI_x_CLOUD_anom"]   = inner_val["ONI"] * inner_val["CLOUD_anom"]

X_inner_tr = inner_train[FINAL_FEATURES].values
y_inner_tr = inner_train[TARGET].values
X_inner_val = inner_val[FINAL_FEATURES].values
y_inner_val = inner_val[TARGET].values

# Conservative parameter grid
PARAM_GRID = {
    "max_depth":         [3, 4, 5],
    "n_estimators":      [50, 100, 150],
    "learning_rate":     [0.03, 0.05, 0.10],
    "subsample":         [0.7, 0.8],
    "colsample_bytree":  [0.7, 0.8],
    "min_child_weight":  [5, 10],   # High value → prevents over-fitting small leaf nodes
}

grid = list(ParameterGrid(PARAM_GRID))
print(f"  Grid size: {len(grid)} combinations")

best_val_rmse  = np.inf
best_params    = None
search_results = []

for params in grid:
    model = xgb.XGBRegressor(
        **params,
        reg_alpha=0.1,     # L1 regularization
        reg_lambda=1.0,    # L2 regularization
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_inner_tr, y_inner_tr,
              eval_set=[(X_inner_val, y_inner_val)],
              verbose=False)

    val_pred = model.predict(X_inner_val)
    val_rmse = np.sqrt(mean_squared_error(y_inner_val, val_pred))

    search_results.append({**params, "val_RMSE": round(val_rmse, 6)})
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params   = params.copy()

df_search = pd.DataFrame(search_results).sort_values("val_RMSE")
print(f"\n  Best params (val RMSE={best_val_rmse:.6f}):")
for k, v in best_params.items():
    print(f"    {k:<22}: {v}")

# Verify: max_depth ≤ 5 enforced
assert best_params.get("max_depth", 0) <= 5, "max_depth exceeded constraint"
print(f"  ✓ max_depth constraint satisfied")

# Q1 AUDIT FIX: persist best_params so notebook 08's cross-fold SHAP
# stability check (also added as part of this audit fix) can reuse the
# SAME hyperparameters, rather than falling back to generic defaults.
import json
with open(f"{OUT_DIR}/07_xgboost_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print(f"  Saved: outputs/07_xgboost_best_params.json")

# ══════════════════════════════════════════════════════════════════════════
# PART B: WALK-FORWARD EVALUATION (9 folds)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: WALK-FORWARD EVALUATION")
print("=" * 60)

"""
For XGBoost walk-forward:
  - Use best_params found in Part A (fixed across all folds — conservative)
  - No re-tuning per fold (would be computationally expensive and
    potentially over-optimistic with only 9 folds)
  - Bootstrap PI: fit model B=200 times with different random seeds
    on training data; use prediction distribution across bootstrap models,
    WITH residual noise added back (see Q1 audit fix note below)
"""

# Q1 AUDIT FIX: was 50 ("Reduced for speed; use 200 for final paper" —
# the code comment's own TODO was never applied before this was
# committed). Set to 200 to match what the manuscript should report.
BOOTSTRAP_SAMPLES = 200

wf_results   = []
all_y_true   = []
all_y_pred   = []
all_pi_lower = []
all_pi_upper = []

for fold_idx, test_year in enumerate(range(2015, 2024)):
    # Q1 AUDIT FIX: real leakage guard for this fold.
    verify_no_feature_leakage(df, test_year, RAW_ANOMALY_SOURCE_COLS)

    X_tr, y_tr, X_te, y_te, dates_te, y_clim_pred = get_split_data(
        df, test_year, features=FINAL_FEATURES,
        raw_anomaly_source_cols=RAW_ANOMALY_SOURCE_COLS, target=TARGET
    )

    # Primary model
    model_fold = xgb.XGBRegressor(
        **best_params,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_fold.fit(X_tr, y_tr)
    y_pred = model_fold.predict(X_te)

    # Q1 AUDIT FIX: training-residual standard deviation, used to add
    # aleatoric (irreducible) noise back into each bootstrap prediction
    # below. The original implementation resampled training data and
    # refit the model B times WITHOUT adding any residual noise term,
    # capturing only parameter/estimation uncertainty. This structurally
    # under-covers the true predictive distribution and is the root
    # cause of the severely miscalibrated PICP reported for XGBoost in
    # the manuscript (excluded from main reporting there) — this fix
    # addresses the root cause directly rather than only working around
    # it downstream.
    train_resid_std = np.std(y_tr - model_fold.predict(X_tr))

    # Bootstrap PI (resample training data with replacement)
    boot_preds = np.zeros((BOOTSTRAP_SAMPLES, len(X_te)))
    for b in range(BOOTSTRAP_SAMPLES):
        rng   = np.random.default_rng(b)
        idx_b = rng.choice(len(X_tr), size=len(X_tr), replace=True)
        m_b   = xgb.XGBRegressor(
            **best_params,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=b, n_jobs=-1, verbosity=0
        )
        m_b.fit(X_tr[idx_b], y_tr[idx_b])
        point_pred_b = m_b.predict(X_te)
        # Add back aleatoric noise (Q1 audit fix) so the bootstrap
        # distribution reflects predictive, not just parameter,
        # uncertainty.
        boot_preds[b] = point_pred_b + rng.normal(0, train_resid_std, size=len(X_te))

    # 90% prediction interval from bootstrap distribution
    pi_lo = np.percentile(boot_preds, 5, axis=0)
    pi_hi = np.percentile(boot_preds, 95, axis=0)

    # Metrics
    fold_rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
    fold_mae  = np.mean(np.abs(y_te - y_pred))
    # Q1 AUDIT FIX: SkillScore vs leakage-free per-fold expanding-window
    # climatological baseline (y_clim_pred from get_split_data()),
    # replacing the original np.std(y_te) test-set-derived formula.
    clim_rmse = np.sqrt(np.mean((y_te - y_clim_pred) ** 2))
    fold_ss   = 1 - fold_rmse / clim_rmse if clim_rmse > 0 else np.nan
    picp_val  = np.mean((y_te >= pi_lo) & (y_te <= pi_hi))

    # Overfitting check: train RMSE vs test RMSE
    y_pred_tr  = model_fold.predict(X_tr)
    train_rmse = np.sqrt(np.mean((y_tr - y_pred_tr) ** 2))

    wf_results.append({
        "fold": fold_idx + 1, "test_year": test_year,
        "n_train": len(X_tr),
        "RMSE_train": round(train_rmse, 6),
        "RMSE_test":  round(fold_rmse, 6),
        "overfit_ratio": round(fold_rmse / train_rmse, 3),
        "MAE": round(fold_mae, 6),
        "SkillScore": round(fold_ss, 4),
        "ClimRMSE": round(clim_rmse, 6),
        "PICP_90": round(picp_val, 4),
    })

    all_y_true.extend(y_te)
    all_y_pred.extend(y_pred)
    all_pi_lower.extend(pi_lo)
    all_pi_upper.extend(pi_hi)

    print(f"  Fold {fold_idx+1} ({test_year}): "
          f"RMSE_tr={train_rmse:.5f} | RMSE_te={fold_rmse:.5f} | "
          f"overfit_ratio={fold_rmse/train_rmse:.2f} | "
          f"SS={fold_ss:.3f} | PICP={picp_val:.3f}")

df_wf = pd.DataFrame(wf_results)

print(f"\n  AGGREGATE XGBoost Walk-forward:")
print(f"  Mean RMSE (test)   = {df_wf['RMSE_test'].mean():.6f} ± {df_wf['RMSE_test'].std():.6f}")
print(f"  Mean SkillScore    = {df_wf['SkillScore'].mean():.4f} ± {df_wf['SkillScore'].std():.4f}")
print(f"  Mean PICP (90%)    = {df_wf['PICP_90'].mean():.4f}  [nominal: 0.900]")
print(f"  Mean overfit ratio = {df_wf['overfit_ratio'].mean():.3f}  [>1.5 = concern]")

if df_wf["overfit_ratio"].mean() > 2.0:
    print(f"  ⚠ High overfit ratio detected → increase regularization in paper discussion")
else:
    print(f"  ✓ Overfit ratio acceptable for n=240 training observations")

# ══════════════════════════════════════════════════════════════════════════
# PART C: FULL-SAMPLE FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C: FEATURE IMPORTANCE (Full-sample model)")
print("=" * 60)

# Train on all data (2005-2023) for feature importance analysis
# NOTE: this model is for SHAP analysis only — NOT for evaluation
full_train = df[df["YEAR"] <= 2023].copy()

# Q1 AUDIT FIX: recompute anomaly-derived features using ONLY the
# 2005-2023 climatology (excludes the 2024-2025 holdout, consistent
# with the manuscript's statement that the holdout is "not used in any
# development step"). The original code reused the *_anom columns
# precomputed once on the full 2005-2025 sample in notebook 03, which
# would leak 2024-2025 holdout information into this SHAP model.
for col in RAW_ANOMALY_SOURCE_COLS:
    full_train[f"{col}_anom"] = full_train[col] - full_train.groupby("MONTH")[col].transform("mean")
full_train["ONI_x_CLOUD_anom"] = full_train["ONI"] * full_train["CLOUD_anom"]

model_full = xgb.XGBRegressor(
    **best_params,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0
)
model_full.fit(full_train[FINAL_FEATURES], full_train[TARGET])

# XGBoost built-in importance (gain-based)
fi_gain = model_full.get_booster().get_score(importance_type="gain")
fi_df   = pd.DataFrame([
    {"Feature": f, "Importance_gain": round(v, 4)}
    for f, v in fi_gain.items()
]).sort_values("Importance_gain", ascending=False)

print(f"  XGBoost Feature Importance (Gain):")
print(fi_df.to_string(index=False))

# ── SAVE ──────────────────────────────────────────────────────────────────
df_wf.to_csv(f"{OUT_DIR}/07_xgboost_walkforward_results.csv", index=False)
fi_df.to_csv(f"{OUT_DIR}/07_xgboost_feature_importance.csv", index=False)

# Save the full-sample model for SHAP analysis in notebook 08
import pickle
with open(f"{DATA_DIR}/07_xgboost_full_model.pkl", "wb") as f:
    pickle.dump(model_full, f)

# Save predictions for DM test
test_idx = (df["YEAR"] >= 2015) & (df["YEAR"] <= 2023)
pred_df  = pd.DataFrame({
    "DATE":           df.loc[test_idx, "DATE"].values[:len(all_y_true)],
    "y_true":         all_y_true,
    "y_pred_xgb":     all_y_pred,
    "pi_lower_90":    all_pi_lower,
    "pi_upper_90":    all_pi_upper,
})
pred_df.to_parquet(f"{DATA_DIR}/07_xgboost_predictions.parquet", index=False)

print(f"\n✅ Notebook 07 complete.")
print(f"   Best params: {best_params}")
print(f"   Walk-forward mean RMSE: {df_wf['RMSE_test'].mean():.6f}")
