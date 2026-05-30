"""
=============================================================================
NOTEBOOK 06: SARIMAX + CLIMATE TELECONNECTION MODEL
=============================================================================
Purpose  : Fit SARIMAX with ONI/DMI exogenous regressors.
           Grid-search optimal (p,d,q)(P,D,Q)12 order via AIC/BIC.
           Walk-forward evaluation with 12-month ahead forecasting.s
           ENSO phase-conditioned performance analysis.

Input    : data/03_model_ready.parquet
           data/03_final_features.csv
Output   : outputs/06_sarimax_order_selection.csv
           outputs/06_sarimax_walkforward_results.csv
           outputs/06_enso_phase_analysis.csv
           data/06_sarimax_predictions.parquet

=============================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools, warnings, os
warnings.filterwarnings("ignore")

DATA_DIR = "/home/claude/pv_research/data"
OUT_DIR  = "/home/claude/pv_research/outputs"

df = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
df = df.set_index("DATE").asfreq("MS")   # Monthly Start frequency

TARGET = "Y_stoch"

print(f"Loaded: {len(df)} monthly observations")
print(f"Period: {df.index.min().date()} → {df.index.max().date()}")

# ══════════════════════════════════════════════════════════════════════════
# PART A: SARIMAX ORDER SELECTION (AIC/BIC grid search)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: SARIMAX ORDER SELECTION — AIC/BIC GRID SEARCH")
print("=" * 60)

"""
Grid search over SARIMA(p,d,q)(P,D,Q)12.
Constraints for n=240 training observations:
  p, q ∈ {0, 1, 2}   — non-seasonal AR/MA
  P, Q ∈ {0, 1}      — seasonal AR/MA
  d = 0               — Y_stoch confirmed stationary (ADF tests)
  D = 0               — seasonal unit root unlikely for monthly equatorial data
  Seasonal period s = 12

Exogenous: ONI (primary ENSO driver)
Reason: DMI is collinear with synthetic ONI in current data;
        add DMI when real external data is downloaded.
"""

EXOG_COLS = ["ONI"]   # Add "DMI" when real data available

# Use training data only (2005–2014) for order selection
train_order = df[df.index.year < 2015]
y_order     = train_order[TARGET]
X_order     = train_order[EXOG_COLS]

p_range = [0, 1, 2]
q_range = [0, 1, 2]
P_range = [0, 1]
Q_range = [0, 1]

grid_results = []

print(f"  Running grid search ({len(p_range)*len(q_range)*len(P_range)*len(Q_range)} candidates)...")

for p, q, P, Q in itertools.product(p_range, q_range, P_range, Q_range):
    try:
        model = SARIMAX(
            y_order,
            exog=X_order,
            order=(p, 0, q),
            seasonal_order=(P, 0, Q, 12),
            enforce_stationarity=True,
            enforce_invertibility=True,
            trend="c"
        )
        res = model.fit(disp=False, maxiter=200)

        # Check residual white noise (Ljung-Box lag 12)
        lb = acorr_ljungbox(res.resid, lags=[12], return_df=True)
        lb_p = lb["lb_pvalue"].values[0]

        grid_results.append({
            "p": p, "q": q, "P": P, "Q": Q,
            "AIC": round(res.aic, 2),
            "BIC": round(res.bic, 2),
            "LB12_p": round(lb_p, 4),
            "LB_ok": lb_p > 0.05,
            "converged": True
        })
    except Exception as e:
        grid_results.append({
            "p": p, "q": q, "P": P, "Q": Q,
            "AIC": np.nan, "BIC": np.nan,
            "LB12_p": np.nan, "LB_ok": False, "converged": False
        })

df_grid = pd.DataFrame(grid_results)
df_grid_valid = df_grid[df_grid["converged"] & df_grid["LB_ok"]].copy()

print(f"\n  Converged + white-noise residuals: {len(df_grid_valid)} / {len(df_grid)} models")
print(f"\n  Top 5 by AIC (valid models only):")
print(df_grid_valid.nsmallest(5, "AIC")[["p","q","P","Q","AIC","BIC","LB12_p"]].to_string(index=False))

# Select best model (AIC-optimal with white-noise residuals)
if len(df_grid_valid) > 0:
    best_row = df_grid_valid.nsmallest(1, "AIC").iloc[0]
    BEST_ORDER    = (int(best_row["p"]), 0, int(best_row["q"]))
    BEST_SEASONAL = (int(best_row["P"]), 0, int(best_row["Q"]), 12)
    print(f"\n  ✓ SELECTED ORDER: SARIMA{BEST_ORDER}{BEST_SEASONAL}")
else:
    # Fallback to literature-informed default
    BEST_ORDER    = (1, 0, 1)
    BEST_SEASONAL = (1, 0, 1, 12)
    print(f"\n  ⚠ No valid models found — using default SARIMA(1,0,1)(1,0,1)12")

df_grid.to_csv(f"{OUT_DIR}/06_sarimax_order_selection.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# PART B: FULL-SAMPLE SARIMAX FIT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: FULL-SAMPLE SARIMAX MODEL")
print("=" * 60)

model_full = SARIMAX(
    df[TARGET],
    exog=df[EXOG_COLS],
    order=BEST_ORDER,
    seasonal_order=BEST_SEASONAL,
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
)
res_full = model_full.fit(disp=False, maxiter=300)

print(f"\n  SARIMAX{BEST_ORDER}{BEST_SEASONAL} + ONI")
print(f"  AIC       = {res_full.aic:.2f}")
print(f"  BIC       = {res_full.bic:.2f}")
print(f"  Log-Lik   = {res_full.llf:.2f}")
print()
print(res_full.summary().tables[1])

# ONI coefficient interpretation
if "ONI" in res_full.params.index:
    oni_coef = res_full.params["ONI"]
    oni_p    = res_full.pvalues["ONI"]
    print(f"\n  ► ONI coefficient: {oni_coef:.5f} (p={oni_p:.4f})")
    if oni_p < 0.05:
        print(f"    → ONI is statistically significant at 5% level")
        print(f"    → Each 1-unit increase in ONI associated with "
              f"{oni_coef:+.5f} kWh/m²/day change in Y_PV")
        print(f"    → El Niño (ONI≈+2): expected Y_PV change = {2*oni_coef:+.5f} kWh/m²/day")
    else:
        print(f"    → ONI not significant at 5% (marginal at {oni_p:.3f}) "
              f"— keep as theoretical prior, report as limitation")

# ══════════════════════════════════════════════════════════════════════════
# PART C: WALK-FORWARD SARIMAX EVALUATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C: WALK-FORWARD EVALUATION (9 folds)")
print("=" * 60)

"""
SARIMAX walk-forward strategy:
  - Refit model on expanding training window each fold
  - Forecast 12 steps ahead (one full year)
  - Use actual ONI values for exogenous forecast (known at prediction time
    for retrospective evaluation; in real deployment, use ONI forecasts)
"""

wf_results = []
all_y_true_sarimax, all_y_pred_sarimax = [], []
all_pi_lower, all_pi_upper = [], []

for fold_idx, test_year in enumerate(range(2015, 2024)):
    train = df[df.index.year < test_year]
    test  = df[df.index.year == test_year]

    y_tr = train[TARGET]
    X_tr = train[EXOG_COLS]
    X_te = test[EXOG_COLS]
    y_te = test[TARGET].values

    try:
        fold_model = SARIMAX(
            y_tr,
            exog=X_tr,
            order=BEST_ORDER,
            seasonal_order=BEST_SEASONAL,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fold_res = fold_model.fit(disp=False, maxiter=200)

        # 12-step ahead forecast with prediction intervals (95%)
        fc = fold_res.get_forecast(steps=12, exog=X_te)
        y_pred = fc.predicted_mean.values
        ci     = fc.conf_int(alpha=0.05)   # 95% PI
        pi_lo  = ci.iloc[:, 0].values
        pi_hi  = ci.iloc[:, 1].values

        # Metrics
        fold_rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
        fold_mae  = np.mean(np.abs(y_te - y_pred))
        fold_ss   = 1 - fold_rmse / (np.std(y_te) if np.std(y_te) > 0 else 1)

        # PICP (95% PI)
        picp_val = np.mean((y_te >= pi_lo) & (y_te <= pi_hi))

        wf_results.append({
            "fold": fold_idx + 1, "test_year": test_year,
            "n_train": len(train),
            "RMSE": round(fold_rmse, 6), "MAE": round(fold_mae, 6),
            "SkillScore": round(fold_ss, 4),
            "PICP_95": round(picp_val, 4),
            "converged": True
        })
        all_y_true_sarimax.extend(y_te)
        all_y_pred_sarimax.extend(y_pred)
        all_pi_lower.extend(pi_lo)
        all_pi_upper.extend(pi_hi)

        print(f"  Fold {fold_idx+1} ({test_year}): RMSE={fold_rmse:.5f} | "
              f"SS={fold_ss:.3f} | PICP={picp_val:.3f}")

    except Exception as e:
        print(f"  Fold {fold_idx+1} ({test_year}): FAILED — {str(e)[:60]}")
        wf_results.append({
            "fold": fold_idx + 1, "test_year": test_year,
            "n_train": len(train), "converged": False,
            "RMSE": np.nan, "MAE": np.nan, "SkillScore": np.nan, "PICP_95": np.nan
        })

df_wf = pd.DataFrame(wf_results)
valid_folds = df_wf[df_wf["converged"]]

print(f"\n  AGGREGATE SARIMAX Walk-forward:")
print(f"  Mean RMSE       = {valid_folds['RMSE'].mean():.6f} ± {valid_folds['RMSE'].std():.6f}")
print(f"  Mean SkillScore = {valid_folds['SkillScore'].mean():.4f} ± {valid_folds['SkillScore'].std():.4f}")
print(f"  Mean PICP (95%) = {valid_folds['PICP_95'].mean():.4f}  [nominal: 0.950]")

# ══════════════════════════════════════════════════════════════════════════
# PART D: ENSO PHASE-CONDITIONED ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART D: ENSO PHASE-CONDITIONED PERFORMANCE")
print("=" * 60)

"""
Key research question: Does ENSO phase significantly affect forecasting accuracy?
If yes → climate-aware forecasting is justified.
"""

# Merge predictions with ENSO phase
test_df = df[(df.index.year >= 2015) & (df.index.year <= 2023)].copy()

if len(all_y_true_sarimax) == len(test_df):
    test_df = test_df.copy()
    test_df["y_pred"]  = all_y_pred_sarimax
    test_df["resid"]   = test_df[TARGET] - test_df["y_pred"]
    test_df["abs_err"] = np.abs(test_df["resid"])

    enso_perf = test_df.groupby("ENSO_phase").agg(
        RMSE      = ("resid",   lambda x: np.sqrt(np.mean(x**2))),
        MAE       = ("abs_err", "mean"),
        mean_GHI  = ("GHI",     "mean"),
        n_months  = ("resid",   "count"),
    ).round(5)

    print(f"\n  ENSO Phase-Conditioned RMSE:")
    print(enso_perf.to_string())
    print(f"\n  Interpretation:")
    if "ElNino" in enso_perf.index and "LaNina" in enso_perf.index:
        en_rmse = enso_perf.loc["ElNino", "RMSE"]
        ln_rmse = enso_perf.loc["LaNina", "RMSE"]
        nu_rmse = enso_perf.loc["Neutral", "RMSE"] if "Neutral" in enso_perf.index else np.nan
        print(f"  El Niño RMSE:  {en_rmse:.5f}")
        print(f"  La Niña RMSE:  {ln_rmse:.5f}")
        print(f"  Neutral RMSE:  {nu_rmse:.5f}")
        if en_rmse != ln_rmse:
            worse = "El Niño" if en_rmse > ln_rmse else "La Niña"
            print(f"  → {worse} periods have higher forecast error → ENSO conditioning valuable")

    enso_perf.to_csv(f"{OUT_DIR}/06_enso_phase_analysis.csv")

# ── SAVE ──────────────────────────────────────────────────────────────────
df_wf.to_csv(f"{OUT_DIR}/06_sarimax_walkforward_results.csv", index=False)

pred_df = pd.DataFrame({
    "DATE":           test_df.index[:len(all_y_true_sarimax)],
    "y_true":         all_y_true_sarimax,
    "y_pred_sarimax": all_y_pred_sarimax,
    "pi_lower_95":    all_pi_lower,
    "pi_upper_95":    all_pi_upper,
})
pred_df.to_parquet(f"{DATA_DIR}/06_sarimax_predictions.parquet", index=False)

print(f"\n✅ Notebook 06 complete.")
print(f"   Best SARIMAX order: {BEST_ORDER}{BEST_SEASONAL}")
print(f"   Walk-forward mean RMSE: {valid_folds['RMSE'].mean():.6f}")
print(f"   PICP 95%: {valid_folds['PICP_95'].mean():.4f} (nominal: 0.950)")
