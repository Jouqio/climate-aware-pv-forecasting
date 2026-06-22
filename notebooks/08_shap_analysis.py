"""
=============================================================================
NOTEBOOK 08: SHAP ANALYSIS + ECONOMETRIC-XAI CORRESPONDENCE
=============================================================================
Purpose  : Compute SHAP values for XGBoost model.
           Build "Econometric-XAI Correspondence" — the paper's novel
           interpretability contribution.
           Compare OLS-HC3 coefficients vs SHAP feature impacts.
           Identify nonlinear effects and ENSO interactions.

Input    : data/07_xgboost_full_model.pkl
           data/03_model_ready.parquet
           outputs/05_ols_coefficients.csv
Output   : outputs/08_shap_values.csv
           outputs/08_shap_feature_summary.csv
           outputs/08_econometric_xai_correspondence.csv
           outputs/08_shap_enso_analysis.csv

=============================================================================
"""

import pandas as pd
import numpy as np
import shap
import pickle
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR.parent))  # utils.py lives at repo root, one level above notebooks/
from utils import expanding_climatology  # noqa: E402

# Load data and models
df = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
FINAL_FEATURES = pd.read_csv(f"{DATA_DIR}/03_final_features.csv")["feature"].tolist()
TARGET = "Y_stoch"
RAW_ANOMALY_SOURCE_COLS = ["GHI", "CLOUD", "PRECTOT", "T2M"]

with open(f"{DATA_DIR}/07_xgboost_full_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

ols_coef = pd.read_csv(f"{OUT_DIR}/05_ols_coefficients.csv")
ols_coef = ols_coef[ols_coef["Feature"] != "const"].copy()

# Use 2005-2023 for SHAP (full training data)
df_analysis = df[df["YEAR"] <= 2023].copy()

# Q1 AUDIT FIX: recompute anomaly-derived features using ONLY 2005-2023
# climatology (excludes the 2024-2025 holdout) — MUST exactly match how
# notebooks/07_xgboost_model.py Part C built `full_train` for the
# pickled xgb_model loaded above. Using a different anomaly computation
# here than what the model was actually trained on would make every
# SHAP value in this notebook attributed to features the model never
# saw in that form — a silent train/explain mismatch, not technically
# "leakage" but a serious correctness bug in its own right.
for col in RAW_ANOMALY_SOURCE_COLS:
    df_analysis[f"{col}_anom"] = df_analysis[col] - df_analysis.groupby("MONTH")[col].transform("mean")
df_analysis["ONI_x_CLOUD_anom"] = df_analysis["ONI"] * df_analysis["CLOUD_anom"]

X_analysis  = df_analysis[FINAL_FEATURES]

print(f"SHAP analysis on {len(df_analysis)} observations")
print(f"Features: {FINAL_FEATURES}")

# ══════════════════════════════════════════════════════════════════════════
# PART A: SHAP VALUE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: SHAP VALUE COMPUTATION (TreeExplainer)")
print("=" * 60)

"""
TreeExplainer for XGBoost:
  - Exact Shapley values (not approximate)
  - Computationally efficient for tree ensembles
  - Consistent with Shapley axioms: efficiency, symmetry, dummy, additivity
  - SHAP value φⱼ(x) = average marginal contribution of feature j
    across all possible feature orderings

For each observation i and feature j:
  SHAP[i,j] = contribution of feature j to prediction i, relative to E[f(X)]
  Sum_j SHAP[i,j] = f(xᵢ) - E[f(X)]   [efficiency property]
"""

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_analysis)  # shape: (n_obs, n_features)
expected_val = explainer.expected_value

print(f"  SHAP base value (E[f(X)]): {expected_val:.5f}")
print(f"  SHAP values shape: {shap_values.shape}")

# Verify efficiency: sum of SHAP ≈ prediction - expected_value
y_pred_full = xgb_model.predict(X_analysis)
shap_sum    = shap_values.sum(axis=1)
efficiency_check = np.allclose(shap_sum + expected_val, y_pred_full, atol=1e-3)
print(f"  ✓ SHAP efficiency property: {efficiency_check}")

# ══════════════════════════════════════════════════════════════════════════
# PART B: SHAP FEATURE SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: SHAP FEATURE SUMMARY")
print("=" * 60)

shap_df = pd.DataFrame(shap_values, columns=FINAL_FEATURES)

# Mean absolute SHAP (importance), mean signed SHAP (direction)
summary = pd.DataFrame({
    "Feature":        FINAL_FEATURES,
    "mean_abs_SHAP":  np.abs(shap_values).mean(axis=0),
    "mean_SHAP":      shap_values.mean(axis=0),
    "std_SHAP":       shap_values.std(axis=0),
    "max_abs_SHAP":   np.abs(shap_values).max(axis=0),
}).sort_values("mean_abs_SHAP", ascending=False).round(6)

print(f"\n  {'Rank':<5} {'Feature':<22} {'mean|SHAP|':>12} {'mean SHAP':>12} "
      f"{'Direction':>12}")
print(f"  {'-'*70}")
for rank, (_, row) in enumerate(summary.iterrows(), 1):
    direction = "↑ Positive" if row["mean_SHAP"] > 0 else "↓ Negative"
    print(f"  {rank:<5} {row['Feature']:<22} {row['mean_abs_SHAP']:>12.6f} "
          f"{row['mean_SHAP']:>12.6f} {direction:>12}")

# ══════════════════════════════════════════════════════════════════════════
# PART C: ECONOMETRIC-XAI CORRESPONDENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C: ECONOMETRIC-XAI CORRESPONDENCE")
print("=" * 60)

"""
NOVEL CONTRIBUTION: Formal comparison of OLS-HC3 coefficients vs SHAP values.

Correspondence metrics:
  1. Sign concordance: do OLS β and SHAP have same direction?
     → Tests whether linear and nonlinear models agree on effect direction
  2. Rank correlation: does feature importance ranking align?
     → Spearman ρ(|β_j|/SE_j, mean|SHAP_j|)
  3. Magnitude ratio: SHAP impact / OLS standardized coefficient
     → Values near 1.0 indicate linear relationship sufficient
     → Values >> 1 or << 1 indicate nonlinearity captured by XGBoost

INTERPRETATION FRAMEWORK:
  - High concordance: OLS is sufficient; XGBoost adds no new physical insight
  - Low concordance on feature X: X has nonlinear effect not captured by OLS
  - Sign flip: OLS is misspecified OR XGBoost is overfitting
"""

# Standardize OLS coefficients by feature std for comparability
feature_stds = df_analysis[FINAL_FEATURES].std()

correspondence = []
for _, ols_row in ols_coef.iterrows():
    feat = ols_row["Feature"]
    if feat not in FINAL_FEATURES:
        continue

    ols_beta = ols_row["Coef"]
    ols_se   = ols_row["HC3_SE"]
    ols_t    = ols_row["t_stat"]
    ols_p    = ols_row["p_value"]

    # SHAP for this feature
    feat_idx   = FINAL_FEATURES.index(feat)
    shap_mean  = shap_values[:, feat_idx].mean()
    shap_abs   = np.abs(shap_values[:, feat_idx]).mean()

    # Standardized OLS: β * std(X) (units: change in Y per std dev of X)
    ols_std_effect = ols_beta * feature_stds.get(feat, 1.0)

    # Sign concordance
    sign_match = (np.sign(ols_beta) == np.sign(shap_mean))

    # Nonlinearity indicator: if |SHAP contribution variance| is high
    # relative to mean, feature has heterogeneous (nonlinear) effect
    shap_cv = shap_df[feat].std() / (abs(shap_df[feat].mean()) + 1e-8)

    correspondence.append({
        "Feature":          feat,
        "OLS_beta":         round(ols_beta, 6),
        "OLS_HC3_SE":       round(ols_se, 6),
        "OLS_t":            round(ols_t, 3),
        "OLS_p":            round(ols_p, 4),
        "OLS_std_effect":   round(ols_std_effect, 6),
        "mean_SHAP":        round(shap_mean, 6),
        "mean_abs_SHAP":    round(shap_abs, 6),
        "SHAP_CV":          round(shap_cv, 3),
        "sign_concordance": sign_match,
        "nonlinearity":     "High" if shap_cv > 2.0 else "Moderate" if shap_cv > 1.0 else "Low",
    })

corr_df = pd.DataFrame(correspondence)

print(f"\n  {'Feature':<22} {'OLS β':>10} {'SHAP mean':>10} {'Sign':>6} "
      f"{'Nonlin.':>10}")
print(f"  {'-'*65}")
for _, row in corr_df.iterrows():
    sign_str = "✓" if row["sign_concordance"] else "✗ FLIP"
    print(f"  {row['Feature']:<22} {row['OLS_beta']:>10.5f} "
          f"{row['mean_SHAP']:>10.5f} {sign_str:>6} {row['nonlinearity']:>10}")

n_concordant = corr_df["sign_concordance"].sum()
n_total      = len(corr_df)
print(f"\n  Sign concordance: {n_concordant}/{n_total} features agree in direction")

# Rank correlation between OLS importance (|t-stat|) and SHAP importance
from scipy.stats import spearmanr
ols_ranks  = corr_df["OLS_t"].abs().values
shap_ranks = corr_df["mean_abs_SHAP"].values
spearman_r, spearman_p = spearmanr(ols_ranks, shap_ranks)
print(f"\n  Spearman rank correlation (|OLS t-stat| vs mean|SHAP|):")
print(f"  ρ = {spearman_r:.4f}  (p = {spearman_p:.4f})")
if spearman_p < 0.05:
    if spearman_r > 0.7:
        print(f"  → HIGH concordance: OLS and XGBoost agree on feature ranking")
        print(f"  → Linear model sufficient for feature importance interpretation")
    else:
        print(f"  → MODERATE concordance: XGBoost reveals nonlinear patterns")
else:
    print(f"  → LOW concordance (not significant): XGBoost and OLS identify different drivers")
    print(f"  → Strong evidence of nonlinearity in data")

# ══════════════════════════════════════════════════════════════════════════
# PART D: ENSO-STRATIFIED SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART D: ENSO-STRATIFIED SHAP ANALYSIS")
print("=" * 60)

"""
Key question: Does the SHAP impact of CLOUD_anom change across ENSO phases?
If yes: the feature is context-dependent → ENSO conditioning is justified.
"""

enso_phases = df_analysis["ENSO_phase"].values
oni_feature_idx = FINAL_FEATURES.index("ONI") if "ONI" in FINAL_FEATURES else None
cloud_feature_idx = FINAL_FEATURES.index("CLOUD_anom") if "CLOUD_anom" in FINAL_FEATURES else None

enso_shap_rows = []
for phase in ["ElNino", "Neutral", "LaNina"]:
    mask = enso_phases == phase
    if mask.sum() == 0:
        continue

    phase_shap_abs = np.abs(shap_values[mask]).mean(axis=0)
    top_features = np.argsort(phase_shap_abs)[::-1][:5]

    enso_shap_rows.append({
        "ENSO_phase":     phase,
        "n_months":       mask.sum(),
        "top1_feature":   FINAL_FEATURES[top_features[0]],
        "top2_feature":   FINAL_FEATURES[top_features[1]],
        "top3_feature":   FINAL_FEATURES[top_features[2]],
        "ONI_SHAP_mean":  round(shap_values[mask, oni_feature_idx].mean(), 6) if oni_feature_idx else np.nan,
        "CLOUD_SHAP_mean": round(shap_values[mask, cloud_feature_idx].mean(), 6) if cloud_feature_idx else np.nan,
    })

df_enso_shap = pd.DataFrame(enso_shap_rows)
print(df_enso_shap.to_string(index=False))

if cloud_feature_idx is not None and oni_feature_idx is not None:
    en_mask = enso_phases == "ElNino"
    ln_mask = enso_phases == "LaNina"
    en_cloud_shap = shap_values[en_mask, cloud_feature_idx].mean() if en_mask.sum() > 0 else np.nan
    ln_cloud_shap = shap_values[ln_mask, cloud_feature_idx].mean() if ln_mask.sum() > 0 else np.nan
    print(f"\n  CLOUD_anom SHAP during El Niño:  {en_cloud_shap:.6f}")
    print(f"  CLOUD_anom SHAP during La Niña:  {ln_cloud_shap:.6f}")
    if not np.isnan(en_cloud_shap) and not np.isnan(ln_cloud_shap):
        if abs(en_cloud_shap - ln_cloud_shap) > 0.001:
            print(f"  → CLOUD effect differs by ENSO phase → nonlinear ENSO-cloud interaction confirmed")
        else:
            print(f"  → CLOUD effect similar across ENSO phases → additive model sufficient")

# ══════════════════════════════════════════════════════════════════════════
# CROSS-FOLD SHAP STABILITY CHECK (Q1 AUDIT FIX — previously missing)
# ══════════════════════════════════════════════════════════════════════════
"""
Q1 CODE AUDIT FIX (2026-06-20): The manuscript (Figure 10B, Table S1)
reports SHAP feature-importance stability across three training window
sizes — full-sample (n≈216), fold-1 (n=108, train 2005-2014), and
fold-9 (n=204, train 2005-2022) — as evidence that the full-sample SHAP
ranking above is representative of the walk-forward evaluation context.
This check did NOT exist anywhere in the repository prior to this fix
(see 39_CODE_AUDIT_CRITICAL_FINDINGS.md, finding #8). It is added here.

Each fold-specific model is trained fresh on that fold's training
window ONLY, with anomaly features recomputed from that window's own
climatology (matching the per-fold discipline used throughout the
walk-forward notebooks) — NOT reusing model_full or df_analysis's
2005-2023-wide anomaly columns, which would defeat the purpose of this
stability check.
"""
print("\n" + "=" * 60)
print("CROSS-FOLD SHAP STABILITY CHECK")
print("=" * 60)

import xgboost as xgb

FOLD_DEFINITIONS = {
    "full_sample": df[df["YEAR"] <= 2023].copy(),   # same as df_analysis above
    "fold_1":      df[df["YEAR"] <= 2014].copy(),    # train window of WF fold 1
    "fold_9":      df[df["YEAR"] <= 2022].copy(),    # train window of WF fold 9
}

# Re-use the same best_params XGBoost was tuned with (Part A of NB07).
# Loaded here from the walk-forward results file to avoid hardcoding.
try:
    import json
    with open(f"{OUT_DIR}/07_xgboost_best_params.json") as _f:
        _best_params = json.load(_f)
except FileNotFoundError:
    print("  ⚠ 07_xgboost_best_params.json not found — falling back to the "
          "same XGBRegressor defaults used elsewhere in this notebook's "
          "model_full for consistency.")
    _best_params = xgb_model.get_params()
    _best_params = {k: v for k, v in _best_params.items()
                    if k in ["max_depth", "n_estimators", "learning_rate",
                             "subsample", "colsample_bytree", "min_child_weight"]}

fold_shap_results = {}
for fold_name, fold_df_window in FOLD_DEFINITIONS.items():
    fdf = fold_df_window.copy()
    for col in RAW_ANOMALY_SOURCE_COLS:
        fdf[f"{col}_anom"] = fdf[col] - fdf.groupby("MONTH")[col].transform("mean")
    fdf["ONI_x_CLOUD_anom"] = fdf["ONI"] * fdf["CLOUD_anom"]

    m = xgb.XGBRegressor(**_best_params, reg_alpha=0.1, reg_lambda=1.0,
                          random_state=42, n_jobs=-1, verbosity=0)
    m.fit(fdf[FINAL_FEATURES], fdf[TARGET])

    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(fdf[FINAL_FEATURES])
    mean_abs_shap = np.abs(sv).mean(axis=0)

    fold_shap_results[fold_name] = pd.Series(mean_abs_shap, index=FINAL_FEATURES)
    print(f"  {fold_name:<12} (n={len(fdf):3d}): "
          f"top feature = {FINAL_FEATURES[np.argmax(mean_abs_shap)]} "
          f"(mean|SHAP|={mean_abs_shap.max():.5f})")

fold_shap_df = pd.DataFrame(fold_shap_results)
fold_shap_df["rank_full_sample"] = fold_shap_df["full_sample"].rank(ascending=False).astype(int)
fold_shap_df["rank_fold_1"]      = fold_shap_df["fold_1"].rank(ascending=False).astype(int)
fold_shap_df["rank_fold_9"]      = fold_shap_df["fold_9"].rank(ascending=False).astype(int)
fold_shap_df = fold_shap_df.sort_values("rank_full_sample")

print("\n  Rank-1 feature stability across training window sizes:")
top_feature_per_window = {
    name: fold_shap_df[f"{name}" if name != "full_sample" else "full_sample"].idxmax()
    for name in FOLD_DEFINITIONS
}
for name, feat in top_feature_per_window.items():
    print(f"    {name:<12}: rank-1 = {feat}")
all_same_top1 = len(set(top_feature_per_window.values())) == 1
print(f"\n  Rank-1 feature IDENTICAL across all 3 window sizes: {all_same_top1}")
if not all_same_top1:
    print("  ⚠ WARNING: the manuscript's stability claim does NOT hold for "
          "this run — update Figure 10B / Table S1 / Section 4.5 text "
          "accordingly before resubmission.")

fold_shap_df.round(5).to_csv(f"{OUT_DIR}/08_shap_fold_stability.csv")
print(f"\n  Saved: outputs/08_shap_fold_stability.csv")

# ── SAVE ──────────────────────────────────────────────────────────────────
shap_df.to_csv(f"{OUT_DIR}/08_shap_values.csv", index=False)
summary.to_csv(f"{OUT_DIR}/08_shap_feature_summary.csv", index=False)
corr_df.to_csv(f"{OUT_DIR}/08_econometric_xai_correspondence.csv", index=False)
df_enso_shap.to_csv(f"{OUT_DIR}/08_shap_enso_analysis.csv", index=False)

print(f"\n✅ Notebook 08 complete.")
print(f"   Sign concordance: {n_concordant}/{n_total}")
print(f"   Spearman ρ (OLS vs SHAP ranking): {spearman_r:.4f}  (p={spearman_p:.4f})")
