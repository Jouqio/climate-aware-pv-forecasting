"""
=============================================================================
NOTEBOOK 07: XGBOOST MODEL
=============================================================================
Purpose  : Constrained XGBoost for monthly PV forecasting.
           Deliberately conservative hyperparameters for n=240 training obs.
           Walk-forward evaluation with IDENTICAL splits as OLS/SARIMAX.ss
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
import os, warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

DATA_DIR = "/home/claude/pv_research/data"
OUT_DIR  = "/home/claude/pv_research/outputs"

df = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
FINAL_FEATURES = pd.read_csv(f"{DATA_DIR}/03_final_features.csv")["feature"].tolist()
TARGET = "Y_stoch"

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
inner_train = df[df["YEAR"] <= 2012]
inner_val   = df[(df["YEAR"] >= 2013) & (df["YEAR"] <= 2014)]

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
  - Bootstrap PI: fit model B=100 times with different random seeds
    on training data; use prediction distribution across bootstrap models
"""

BOOTSTRAP_SAMPLES = 50   # Reduced for speed; use 200 for final paper

wf_results   = []
all_y_true   = []
all_y_pred   = []
all_pi_lower = []
all_pi_upper = []

for fold_idx, test_year in enumerate(range(2015, 2024)):
    train = df[df["YEAR"] < test_year]
    test  = df[df["YEAR"] == test_year]

    X_tr = train[FINAL_FEATURES].values
    y_tr = train[TARGET].values
    X_te = test[FINAL_FEATURES].values
    y_te = test[TARGET].values

    # Primary model
    model_fold = xgb.XGBRegressor(
        **best_params,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_fold.fit(X_tr, y_tr)
    y_pred = model_fold.predict(X_te)

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
        boot_preds[b] = m_b.predict(X_te)

    # 90% prediction interval from bootstrap distribution
    pi_lo = np.percentile(boot_preds, 5, axis=0)
    pi_hi = np.percentile(boot_preds, 95, axis=0)

    # Metrics
    fold_rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
    fold_mae  = np.mean(np.abs(y_te - y_pred))
    fold_ss   = 1 - fold_rmse / (np.std(y_te) if np.std(y_te) > 0 else 1)
    picp_val  = np.mean((y_te >= pi_lo) & (y_te <= pi_hi))

    # Overfitting check: train RMSE vs test RMSE
    y_pred_tr  = model_fold.predict(X_tr)
    train_rmse = np.sqrt(np.mean((y_tr - y_pred_tr) ** 2))

    wf_results.append({
        "fold": fold_idx + 1, "test_year": test_year,
        "n_train": len(train),
        "RMSE_train": round(train_rmse, 6),
        "RMSE_test":  round(fold_rmse, 6),
        "overfit_ratio": round(fold_rmse / train_rmse, 3),
        "MAE": round(fold_mae, 6),
        "SkillScore": round(fold_ss, 4),
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
full_train = df[df["YEAR"] <= 2023]
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
