"""
=============================================================================
NOTEBOOK 09: RESIDUAL DIAGNOSTICS + MODEL COMPARISON
=============================================================================
Purpose  : Cross-model residual analysis and statistical comparison.
           Diebold-Mariano pairwise tests.
           Friedman multi-model ranking test.
           ENSO-residual linkage analysis.
           Structural break examination of residuals.

Input    : data/05_ols_predictions.parquet
           data/06_sarimax_predictions.parquet
           data/07_xgboost_predictions.parquet
           data/03_model_ready.parquet
Output   : outputs/09_model_comparison_table.csv
           outputs/09_diebold_mariano_matrix.csv
           outputs/09_friedman_test_results.csv
           outputs/09_enso_residual_analysis.csv
           outputs/09_residual_acf_stats.csv

=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import warnings, os, sys
from pathlib import Path
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR.parent))  # utils.py lives at repo root, one level above notebooks/
from utils import climatology_baseline_predict  # noqa: E402

# ── Load all predictions ───────────────────────────────────────────────────
df_meta  = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
df_ols   = pd.read_parquet(f"{DATA_DIR}/05_ols_predictions.parquet")
df_sar   = pd.read_parquet(f"{DATA_DIR}/06_sarimax_predictions.parquet")
df_xgb   = pd.read_parquet(f"{DATA_DIR}/07_xgboost_predictions.parquet")

# Align on common test period 2015–2023
# All three notebooks produce predictions for the same test observations
# (same walk-forward splits → same DATE index)
min_len = min(len(df_ols), len(df_sar), len(df_xgb))
df_ols  = df_ols.iloc[:min_len].reset_index(drop=True)
df_sar  = df_sar.iloc[:min_len].reset_index(drop=True)
df_xgb  = df_xgb.iloc[:min_len].reset_index(drop=True)

# Merge into single comparison frame
df_cmp = pd.DataFrame({
    "DATE":       df_ols["DATE"],
    "y_true":     df_ols["y_true"],
    "y_ols":      df_ols["y_pred_ols"],
    "y_sarimax":  df_sar["y_pred_sarimax"],
    "y_xgb":      df_xgb["y_pred_xgb"],
})

# Merge ENSO phase from meta
meta_sub = df_meta[df_meta["YEAR"].between(2015, 2023)][["DATE","ENSO_phase","ONI","GHI"]].copy()
df_cmp   = df_cmp.merge(meta_sub, on="DATE", how="left")

# Compute residuals
df_cmp["resid_ols"]    = df_cmp["y_true"] - df_cmp["y_ols"]
df_cmp["resid_sarimax"]= df_cmp["y_true"] - df_cmp["y_sarimax"]
df_cmp["resid_xgb"]    = df_cmp["y_true"] - df_cmp["y_xgb"]

print(f"Comparison dataset: {len(df_cmp)} observations ({df_cmp['DATE'].min().date()} → {df_cmp['DATE'].max().date()})")

# ══════════════════════════════════════════════════════════════════════════
# PART A: AGGREGATE MODEL PERFORMANCE TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: AGGREGATE PERFORMANCE TABLE")
print("=" * 60)

def compute_metrics(y_true, y_pred, model_name):
    """Full metric suite for one model."""
    e    = y_true - y_pred
    rmse = np.sqrt(np.mean(e ** 2))
    mae  = np.mean(np.abs(e))
    mape = np.mean(np.abs(e / y_true)) * 100
    r2   = 1 - np.sum(e**2) / np.sum((y_true - y_true.mean())**2)
    ss   = 1 - rmse / np.sqrt(np.mean((y_true - y_true.mean())**2))

    # Residual normality
    _, jb_p  = stats.jarque_bera(e)[:2]
    _, sw_p  = stats.shapiro(e[:50])

    # Residual autocorrelation (Ljung-Box lag 1 and 12)
    lb = acorr_ljungbox(e, lags=[1, 12], return_df=True)
    lb1_p  = lb["lb_pvalue"].iloc[0]
    lb12_p = lb["lb_pvalue"].iloc[1]

    # ADF on residuals (should be stationary → white noise)
    adf_stat, adf_p, *_ = adfuller(e, autolag="AIC")

    return {
        "Model":       model_name,
        "RMSE":        round(rmse, 6),
        "MAE":         round(mae, 6),
        "MAPE_%":      round(mape, 4),
        "R2":          round(r2, 4),
        "SkillScore":  round(ss, 4),
        "JB_p":        round(jb_p, 4),
        "SW_p":        round(sw_p, 4),
        "LB1_p":       round(lb1_p, 4),
        "LB12_p":      round(lb12_p, 4),
        "ADF_p_resid": round(adf_p, 4),
        "Resid_norm":  "Normal" if jb_p > 0.05 else "Non-normal",
        "Resid_AC1":   "No AC" if lb1_p > 0.05 else "AC present",
        "Resid_AC12":  "No SAC" if lb12_p > 0.05 else "Seasonal AC",
    }

y_true = df_cmp["y_true"].values
perf_rows = [
    compute_metrics(y_true, df_cmp["y_ols"].values,    "OLS-HC3"),
    compute_metrics(y_true, df_cmp["y_sarimax"].values, "SARIMAX+ONI"),
    compute_metrics(y_true, df_cmp["y_xgb"].values,    "XGBoost"),
]

# ──────────────────────────────────────────────────────────────────────────
# Climatology baseline — Q1 AUDIT FIX (CRITICAL, finding #4)
# ──────────────────────────────────────────────────────────────────────────
"""
The original implementation computed:
    y_clim = df_cmp.groupby(df_cmp["DATE"].dt.month)["y_true"].transform("mean")
i.e., the calendar-month mean of Y_stoch using ALL of 2015-2023 test-
period data COMBINED. This means, e.g., the "baseline forecast" for
January 2015 used the average of January 2015 AND January 2016-2023 —
using six to eight YEARS OF FUTURE DATA to predict the earliest test
year. This is precisely the "aggregate baseline" methodological error
that the manuscript's Methods Section 3.5 explicitly identifies and
claims to correct via a per-fold expanding-window baseline — yet this
exact bug was still present in this notebook's final comparison table
(see 39_CODE_AUDIT_CRITICAL_FINDINGS.md, finding #4).

The corrected version below builds the climatology baseline by
concatenating each fold's OWN train-only (expanding-window) prediction,
using utils.climatology_baseline_predict(), then merges it onto df_cmp
by DATE to guarantee correct row alignment regardless of df_cmp's
internal ordering.
"""
y_clim_parts = []
for test_year in range(2015, 2024):
    train_y = df_meta[df_meta["YEAR"] < test_year]
    test_y  = df_meta[df_meta["YEAR"] == test_year][["DATE", "MONTH"]].copy()
    test_y["y_clim"] = climatology_baseline_predict(train_y, test_y, "Y_stoch")
    y_clim_parts.append(test_y[["DATE", "y_clim"]])

df_clim = pd.concat(y_clim_parts, ignore_index=True)
df_cmp = df_cmp.merge(df_clim, on="DATE", how="left")
assert df_cmp["y_clim"].isna().sum() == 0, \
    "Climatology baseline merge failed for some dates — check DATE alignment."
y_clim = df_cmp["y_clim"].values

perf_rows.append(compute_metrics(y_true, y_clim, "Climatology_baseline"))

df_perf = pd.DataFrame(perf_rows)
print(df_perf[["Model","RMSE","MAE","MAPE_%","R2","SkillScore"]].to_string(index=False))

print(f"\n  Best model by RMSE: {df_perf.nsmallest(1,'RMSE')['Model'].values[0]}")
print(f"  Best model by SkillScore: {df_perf.nlargest(1,'SkillScore')['Model'].values[0]}")

print(f"\n  Residual diagnostics summary:")
print(df_perf[["Model","Resid_norm","Resid_AC1","Resid_AC12","ADF_p_resid"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# PART B: DIEBOLD-MARIANO PAIRWISE TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: DIEBOLD-MARIANO PAIRWISE TESTS")
print("=" * 60)

"""
DM test: H0: Equal predictive accuracy.
         H1: Model A significantly better than Model B.
         H1 (one-sided): negative DM stat → Model A better.
         Harvey-Leybourne-Newbold small-sample correction applied.

Reference: Diebold & Mariano (1995); Harvey et al. (1997).
"""

def diebold_mariano(e1, e2, h=1, crit="mse"):
    """
    e1, e2: forecast errors (not squared) for model 1 and model 2.
    H0: Model 1 and Model 2 equally accurate.
    DM < 0 and significant: Model 1 BETTER than Model 2.
    """
    if crit == "mse":
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) - np.abs(e2)

    n    = len(d)
    dbar = np.mean(d)
    # Newey-West variance with HLN correction
    gamma0 = np.var(d, ddof=1)
    V_d    = gamma0 * (n + 1 - 2*h + h*(h-1)/n) / n
    dm_stat = dbar / np.sqrt(max(V_d, 1e-12))
    p_two   = 2 * stats.t.sf(abs(dm_stat), df=n-1)
    return round(dm_stat, 4), round(p_two, 4)

models  = ["OLS-HC3", "SARIMAX+ONI", "XGBoost"]
errors  = {
    "OLS-HC3":     df_cmp["resid_ols"].values,
    "SARIMAX+ONI": df_cmp["resid_sarimax"].values,
    "XGBoost":     df_cmp["resid_xgb"].values,
}

dm_records = []
print(f"\n  {'Model A':<16} vs {'Model B':<16} {'DM stat':>10} {'p-value':>10} {'Conclusion'}")
print(f"  {'-'*75}")

for m_a in models:
    for m_b in models:
        if m_a == m_b:
            continue
        dm_stat, dm_p = diebold_mariano(errors[m_a], errors[m_b])
        conclusion = (
            f"A better (5%)" if dm_stat < 0 and dm_p < 0.05 else
            f"A better (10%)" if dm_stat < 0 and dm_p < 0.10 else
            "No difference"
        )
        dm_records.append({
            "Model_A": m_a, "Model_B": m_b,
            "DM_stat": dm_stat, "p_value": dm_p,
            "Conclusion": conclusion
        })
        print(f"  {m_a:<16}    {m_b:<16} {dm_stat:>10.4f} {dm_p:>10.4f}  {conclusion}")

df_dm = pd.DataFrame(dm_records)

# ══════════════════════════════════════════════════════════════════════════
# PART C: FRIEDMAN RANKING TEST (multi-model non-parametric)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C: FRIEDMAN MULTI-MODEL RANKING TEST")
print("=" * 60)

"""
Friedman test: non-parametric; tests if models differ significantly
in rank across the 9 walk-forward folds.
H0: All models have equal predictive accuracy (equal mean ranks).
If rejected → follow up with Nemenyi post-hoc pairwise test.
"""

# Load fold-level RMSE from each model's walk-forward results
ols_wf = pd.read_csv(f"{OUT_DIR}/05_ols_walkforward_results.csv")
sar_wf = pd.read_csv(f"{OUT_DIR}/06_sarimax_walkforward_results.csv")
xgb_wf = pd.read_csv(f"{OUT_DIR}/07_xgboost_walkforward_results.csv")

# Align valid folds
valid_sar = sar_wf[sar_wf.get("converged", True) == True]
n_folds   = min(len(ols_wf), len(valid_sar), len(xgb_wf))

rmse_ols = ols_wf["RMSE"].values[:n_folds]
rmse_sar = valid_sar["RMSE"].values[:n_folds]
rmse_xgb = xgb_wf["RMSE_test"].values[:n_folds]

print(f"  Using {n_folds} walk-forward folds for Friedman test")
print(f"\n  Fold-level RMSE:")
print(f"  {'Fold':<6} {'OLS-HC3':>12} {'SARIMAX+ONI':>14} {'XGBoost':>12}")
for i in range(n_folds):
    print(f"  {i+1:<6} {rmse_ols[i]:>12.6f} {rmse_sar[i]:>14.6f} {rmse_xgb[i]:>12.6f}")

# Friedman statistic
data_matrix = np.column_stack([rmse_ols, rmse_sar, rmse_xgb])  # (n_folds, 3)
ranks       = np.array([stats.rankdata(row) for row in data_matrix])  # rank per fold
mean_ranks  = ranks.mean(axis=0)
k           = 3
n_f         = n_folds

chi2_friedman = (12 * n_f / (k * (k + 1))) * (
    np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4
)
p_friedman = stats.chi2.sf(chi2_friedman, df=k - 1)

print(f"\n  Friedman test:")
print(f"  Chi² = {chi2_friedman:.4f}  (df={k-1})")
print(f"  p    = {p_friedman:.4f}")
print(f"  Mean ranks: OLS={mean_ranks[0]:.3f} | SARIMAX={mean_ranks[1]:.3f} | XGBoost={mean_ranks[2]:.3f}")
model_names = ["OLS-HC3", "SARIMAX+ONI", "XGBoost"]
best_model  = model_names[np.argmin(mean_ranks)]

if p_friedman < 0.05:
    print(f"  → Significant difference detected (p<0.05)")
    print(f"  → Best ranked model: {best_model}")
    # Nemenyi critical difference (α=0.05)
    q_alpha     = 2.343   # Nemenyi q-critical value for k=3, α=0.05
    cd          = q_alpha * np.sqrt(k * (k + 1) / (6 * n_f))
    print(f"\n  Nemenyi post-hoc critical difference (α=0.05): CD = {cd:.3f}")
    print(f"  Pairwise differences:")
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i >= j:
                continue
            diff = abs(mean_ranks[i] - mean_ranks[j])
            sig  = "Significant" if diff > cd else "Not significant"
            print(f"    |{m1} - {m2}| = {diff:.3f}  →  {sig}")
else:
    print(f"  → No significant difference (p={p_friedman:.4f} ≥ 0.05)")
    print(f"  → All three models perform equivalently → OLS-HC3 preferred (parsimony)")

# ══════════════════════════════════════════════════════════════════════════
# PART D: ENSO-RESIDUAL LINKAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART D: ENSO-RESIDUAL LINKAGE ANALYSIS")
print("=" * 60)

"""
Key scientific question:
  Do models systematically over/under-predict during ENSO events?
  If yes → residual pattern linked to unmodeled ENSO dynamics.

Test:
  1. Compare mean residuals by ENSO phase
  2. Kruskal-Wallis test: H0 = residual distribution same across phases
  3. If significant → ENSO phase is a confounding variable
"""

for model_name, resid_col in [("OLS-HC3","resid_ols"),
                                ("SARIMAX+ONI","resid_sarimax"),
                                ("XGBoost","resid_xgb")]:
    print(f"\n  Model: {model_name}")
    phase_resid = df_cmp.groupby("ENSO_phase")[resid_col].agg(
        mean="mean", std="std", n="count"
    ).round(6)
    print(phase_resid.to_string())

    # Kruskal-Wallis test
    groups = [df_cmp.loc[df_cmp["ENSO_phase"]==p, resid_col].values
              for p in ["ElNino","LaNina","Neutral"] if p in df_cmp["ENSO_phase"].values]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        if kw_p < 0.05:
            print(f"  Kruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.4f} → "
                  f"Residuals DIFFER by ENSO phase → unmodeled ENSO effect")
        else:
            print(f"  Kruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.4f} → "
                  f"Residuals similar across ENSO phases → model captures ENSO effect")

# Cross-correlation: ONI vs residuals (is there a lagged ENSO signal in residuals?)
print(f"\n  Cross-correlation: ONI(t-k) × OLS residuals(t)")
print(f"  {'Lag':>5} {'Correlation':>14} {'Interpretation'}")
for lag in range(0, 7):
    oni_lagged = df_cmp["ONI"].shift(lag).values
    resid_ols  = df_cmp["resid_ols"].values
    mask       = ~np.isnan(oni_lagged)
    r, p       = stats.pearsonr(oni_lagged[mask], resid_ols[mask])
    sig        = " *" if p < 0.05 else ""
    interp     = "ENSO unexplained variance" if abs(r) > 0.15 and p < 0.05 else "Negligible"
    print(f"  {lag:>5} {r:>14.4f}{sig}  {interp}")

# ── SAVE ──────────────────────────────────────────────────────────────────
df_perf.to_csv(f"{OUT_DIR}/09_model_comparison_table.csv", index=False)
df_dm.to_csv(f"{OUT_DIR}/09_diebold_mariano_matrix.csv", index=False)

friedman_df = pd.DataFrame({
    "Model":      model_names,
    "MeanRank":   mean_ranks.round(3),
    "FriedmanChi2": [chi2_friedman]*3,
    "Friedman_p": [p_friedman]*3,
})
friedman_df.to_csv(f"{OUT_DIR}/09_friedman_test_results.csv", index=False)

print(f"\n✅ Notebook 09 complete.")
print(f"   Best model by aggregate RMSE: {df_perf.nsmallest(1,'RMSE')['Model'].values[0]}")
print(f"   Best model by Friedman rank: {best_model}")
print(f"   DM test results saved to outputs/09_diebold_mariano_matrix.csv")
