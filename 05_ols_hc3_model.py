"""
=============================================================================
NOTEBOOK 05: OLS-HC3 ECONOMETRIC BASELINE
=============================================================================
Purpose  : Full econometric OLS pipeline with HC3 robust standard errors,
           formal hypothesis testing, and complete diagnostic suite.
           This model provides: (a) interpretable coefficients, (b) formal
           statistical inference, (c) econometric hardening evidence,
           (d) baseline for Diebold-Mariano comparison.s

Input    : data/03_model_ready.parquet
           data/03_final_features.csv
Output   : outputs/05_ols_coefficients.csv
           outputs/05_ols_diagnostics.csv
           outputs/05_ols_walkforward_results.csv

=============================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_white, het_breuschpagan, acorr_breusch_godfrey,
    acorr_ljungbox
)
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/claude/pv_research/data"
OUT_DIR  = "/home/claude/pv_research/outputs"

df = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
FINAL_FEATURES = pd.read_csv(f"{DATA_DIR}/03_final_features.csv")["feature"].tolist()
TARGET = "Y_stoch"

print(f"Loaded: {len(df)} observations, {len(FINAL_FEATURES)} features")

# ══════════════════════════════════════════════════════════════════════════
# PART A: FULL-SAMPLE OLS-HC3 MODEL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: FULL-SAMPLE OLS WITH HC3 ROBUST STANDARD ERRORS")
print("=" * 60)

X_full = sm.add_constant(df[FINAL_FEATURES])
y_full = df[TARGET]

# OLS with HC3 (MacKinnon-White heteroskedasticity-robust SE)
# HC3 is preferred for n<250: provides better finite-sample correction
ols_hc3 = sm.OLS(y_full, X_full).fit(cov_type="HC3")

print(f"\n  Model: OLS-HC3 | n={len(df)} | k={len(FINAL_FEATURES)+1}")
print(f"  R²         = {ols_hc3.rsquared:.4f}")
print(f"  Adj. R²    = {ols_hc3.rsquared_adj:.4f}")
print(f"  F-stat     = {ols_hc3.fvalue:.4f}  (p={ols_hc3.f_pvalue:.4e})")
print(f"  AIC        = {ols_hc3.aic:.2f}")
print(f"  BIC        = {ols_hc3.bic:.2f}")

print(f"\n  {'Feature':<22} {'Coef':>10} {'HC3-SE':>10} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
print(f"  {'-'*70}")
for name, coef, se, t, p in zip(
        ols_hc3.params.index,
        ols_hc3.params.values,
        ols_hc3.bse.values,
        ols_hc3.tvalues.values,
        ols_hc3.pvalues.values):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."
    print(f"  {name:<22} {coef:>10.5f} {se:>10.5f} {t:>8.3f} {p:>8.4f} {sig:>5}")

# Save coefficient table
coef_df = pd.DataFrame({
    "Feature":  ols_hc3.params.index,
    "Coef":     ols_hc3.params.values,
    "HC3_SE":   ols_hc3.bse.values,
    "t_stat":   ols_hc3.tvalues.values,
    "p_value":  ols_hc3.pvalues.values,
    "CI_lower": ols_hc3.conf_int()[0].values,
    "CI_upper": ols_hc3.conf_int()[1].values,
}).round(6)

# ══════════════════════════════════════════════════════════════════════════
# PART B: UNIT ROOT TESTS (Pre-modeling)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: UNIT ROOT TESTS (ADF + KPSS)")
print("=" * 60)

test_series = {"Y_stoch": df["Y_stoch"], "GHI": df["GHI"],
               "CLOUD": df["CLOUD"], "ONI": df["ONI"]}
unit_root_results = []

for series_name, series in test_series.items():
    # ADF (H0: unit root exists; reject → stationary)
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(series, autolag="AIC")
    adf_conc = "Stationary" if adf_p < 0.05 else "Non-stationary"

    # KPSS (H0: stationary; reject → non-stationary) — OPPOSITE null
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression="c", nlags="auto")
        kpss_conc = "Stationary" if kpss_p > 0.05 else "Non-stationary"
    except Exception:
        kpss_stat, kpss_p, kpss_conc = np.nan, np.nan, "Error"

    # Consensus
    if adf_conc == kpss_conc == "Stationary":
        consensus = "✓ Stationary"
    elif adf_conc == kpss_conc == "Non-stationary":
        consensus = "✗ Non-stationary → difference"
    else:
        consensus = "~ Ambiguous → check STL"

    unit_root_results.append({
        "Series": series_name,
        "ADF_stat": round(adf_stat, 3), "ADF_p": round(adf_p, 4),
        "ADF_conc": adf_conc,
        "KPSS_stat": round(kpss_stat, 3) if not np.isnan(kpss_stat) else "NA",
        "KPSS_p": round(kpss_p, 4) if not np.isnan(kpss_p) else "NA",
        "KPSS_conc": kpss_conc,
        "Consensus": consensus
    })
    print(f"  {series_name:<12}: ADF p={adf_p:.4f} ({adf_conc}) | "
          f"KPSS p={kpss_p:.4f} ({kpss_conc}) → {consensus}")

# ══════════════════════════════════════════════════════════════════════════
# PART C: RESIDUAL DIAGNOSTIC SUITE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C: RESIDUAL DIAGNOSTICS (Post-estimation)")
print("=" * 60)

resid = ols_hc3.resid.values
y_hat = ols_hc3.fittedvalues.values

diag_results = {}

# 1. Durbin-Watson
dw = durbin_watson(resid)
diag_results["Durbin_Watson"] = {
    "stat": round(dw, 4),
    "interpretation": ("Strong positive autocorr" if dw < 1.5
                       else "No autocorr" if 1.5 <= dw <= 2.5
                       else "Strong negative autocorr")
}

# 2. Breusch-Godfrey (serial correlation, lags 1 and 12)
bg_stat1, bg_p1, _, _ = acorr_breusch_godfrey(ols_hc3, nlags=1)
bg_stat12, bg_p12, _, _ = acorr_breusch_godfrey(ols_hc3, nlags=12)
diag_results["Breusch_Godfrey_lag1"] = {
    "stat": round(bg_stat1, 4), "p_value": round(bg_p1, 4),
    "interpretation": "Serial correlation detected (use Newey-West)" if bg_p1 < 0.05 else "No serial correlation"
}
diag_results["Breusch_Godfrey_lag12"] = {
    "stat": round(bg_stat12, 4), "p_value": round(bg_p12, 4),
    "interpretation": "Seasonal autocorrelation (consider SARIMAX)" if bg_p12 < 0.05 else "No seasonal autocorrelation"
}

# 3. Ljung-Box (lags 1, 6, 12)
lb_results = acorr_ljungbox(resid, lags=[1, 6, 12], return_df=True)
for lag, row in lb_results.iterrows():
    diag_results[f"Ljung_Box_lag{lag}"] = {
        "stat": round(row["lb_stat"], 4), "p_value": round(row["lb_pvalue"], 4),
        "interpretation": "Residual autocorrelation present" if row["lb_pvalue"] < 0.05 else "White noise"
    }

# 4. White test (heteroskedasticity)
white_stat, white_p, white_f, white_fp = het_white(resid, X_full)
diag_results["White_Test"] = {
    "stat": round(white_stat, 4), "p_value": round(white_p, 4),
    "interpretation": "Heteroskedastic → HC3 SE correct choice" if white_p < 0.05 else "Homoskedastic"
}

# 5. Breusch-Pagan
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(resid, X_full)
diag_results["Breusch_Pagan"] = {
    "stat": round(bp_stat, 4), "p_value": round(bp_p, 4),
    "interpretation": "Heteroskedastic" if bp_p < 0.05 else "Homoskedastic"
}

# 6. Jarque-Bera (normality)
jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
diag_results["Jarque_Bera"] = {
    "stat": round(jb_stat, 4), "p_value": round(jb_p, 4),
    "skewness": round(jb_skew, 4), "kurtosis": round(jb_kurt, 4),
    "interpretation": "Non-normal residuals → use bootstrap CI" if jb_p < 0.05 else "Normal residuals"
}

# 7. Shapiro-Wilk (on subsample ≤50)
sw_stat, sw_p = stats.shapiro(resid[:50])
diag_results["Shapiro_Wilk"] = {
    "stat": round(sw_stat, 4), "p_value": round(sw_p, 4),
    "interpretation": "Non-normal" if sw_p < 0.05 else "Normal (consistent with JB)"
}

print(f"\n  {'Test':<30} {'Stat':>10} {'p-value':>10} {'Interpretation'}")
print(f"  {'-'*80}")
for test_name, res in diag_results.items():
    stat_str = f"{res['stat']:.4f}" if "stat" in res else "—"
    p_str    = f"{res['p_value']:.4f}" if "p_value" in res else "—"
    interp   = res.get("interpretation", "—")
    print(f"  {test_name:<30} {stat_str:>10} {p_str:>10}  {interp}")

# ══════════════════════════════════════════════════════════════════════════
# PART D: WALK-FORWARD OLS EVALUATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART D: WALK-FORWARD EVALUATION")
print("=" * 60)

wf_results = []
all_y_true, all_y_pred = [], []

for fold_idx, test_year in enumerate(range(2015, 2024)):
    train = df[df["YEAR"] < test_year]
    test  = df[df["YEAR"] == test_year]

    X_tr = sm.add_constant(train[FINAL_FEATURES], has_constant="add")
    y_tr = train[TARGET]
    X_te = sm.add_constant(test[FINAL_FEATURES], has_constant="add")
    y_te = test[TARGET].values

    # Refit OLS on expanding training window
    fold_ols = sm.OLS(y_tr, X_tr).fit(cov_type="HC3")
    y_pred   = fold_ols.predict(X_te)

    fold_rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
    fold_mae  = np.mean(np.abs(y_te - y_pred))
    fold_ss   = 1 - fold_rmse / np.sqrt(np.mean((y_te - y_te.mean()) ** 2))

    wf_results.append({
        "fold": fold_idx + 1, "test_year": test_year,
        "n_train": len(train), "n_test": len(test),
        "RMSE": round(fold_rmse, 6), "MAE": round(fold_mae, 6),
        "SkillScore": round(fold_ss, 4),
        "R2_train": round(fold_ols.rsquared, 4)
    })
    all_y_true.extend(y_te)
    all_y_pred.extend(y_pred)

    print(f"  Fold {fold_idx+1} ({test_year}): train={len(train):3d} | "
          f"RMSE={fold_rmse:.5f} | MAE={fold_mae:.5f} | SS={fold_ss:.3f}")

df_wf = pd.DataFrame(wf_results)
overall_rmse = np.sqrt(np.mean((np.array(all_y_true) - np.array(all_y_pred)) ** 2))
print(f"\n  AGGREGATE OLS Walk-forward RMSE: {overall_rmse:.6f}")
print(f"  AGGREGATE OLS SkillScore: {df_wf['SkillScore'].mean():.4f} ± {df_wf['SkillScore'].std():.4f}")

# ── SAVE ──────────────────────────────────────────────────────────────────
coef_df.to_csv(f"{OUT_DIR}/05_ols_coefficients.csv", index=False)
df_wf.to_csv(f"{OUT_DIR}/05_ols_walkforward_results.csv", index=False)
pd.DataFrame(list(diag_results.items()), columns=["Test","Results"]).to_csv(
    f"{OUT_DIR}/05_ols_diagnostics.csv", index=False)
pd.DataFrame(unit_root_results).to_csv(f"{OUT_DIR}/05_unit_root_tests.csv", index=False)

# Store predictions for DM test (Notebook 09)
pred_df = pd.DataFrame({
    "DATE": df[df["YEAR"].between(2015, 2023)]["DATE"].values[:len(all_y_true)],
    "y_true": all_y_true, "y_pred_ols": all_y_pred
})
pred_df.to_parquet(f"{DATA_DIR}/05_ols_predictions.parquet", index=False)

print(f"\n✅ Notebook 05 complete.")
print(f"   Full OLS-HC3 model: R²={ols_hc3.rsquared:.4f}, Adj.R²={ols_hc3.rsquared_adj:.4f}")
print(f"   Walk-forward mean RMSE: {df_wf['RMSE'].mean():.6f}")
