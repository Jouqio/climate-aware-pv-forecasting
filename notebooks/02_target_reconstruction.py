"""
=============================================================================
NOTEBOOK 02: STOCHASTIC TARGET RECONSTRUCTION
=============================================================================
Purpose  : Build Y_PV with physics-based stochastic uncertainty components.
           Demonstrate deterministic leakage vs. stochastic target.
           This is the CORE METHODOLOGICAL CONTRIBUTION of the paper.

Input    : data/01_nasa_power_clean.parquet
Output   : data/02_target_reconstructed.parquet
           outputs/02_leakage_demonstration.csv
           outputs/02_stochastic_target_stats.csv

Dependencies : pandas, numpy, scipy, statsmodels
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

np.random.seed(42)   # Reproducibility — document seed in paper

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(f"{DATA_DIR}/01_nasa_power_clean.parquet")
print(f"Loaded: {df.shape[0]} monthly observations, {df.shape[1]} columns")
print(f"Period: {df['DATE'].min().date()} → {df['DATE'].max().date()}")

# ══════════════════════════════════════════════════════════════════════════
# PART A: DEMONSTRATE DETERMINISTIC LEAKAGE (show the old problem)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: DETERMINISTIC LEAKAGE DEMONSTRATION")
print("=" * 60)

"""
Old target construction (PROBLEMATIC):
Y_det = eta_ref * A * GHI * (1 - beta_temp * (T2M - T_ref))

This is algebraically determined by GHI and T2M.
Any model using GHI and T2M as features trivially achieves R²≈0.99
by reconstructing this algebraic identity.
"""

# Parameters for deterministic formula (typical literature values)
ETA_REF   = 0.18    # Reference panel efficiency (monocrystalline Si)
A_PANEL   = 1.0     # Normalized panel area (m²)
BETA_TEMP = 0.004   # Temperature coefficient (/°C), typical -0.4%/°C
T_REF     = 25.0    # STC reference temperature (°C)

# Build DETERMINISTIC target
df["Y_det"] = (ETA_REF * A_PANEL * df["GHI"]
               * (1 - BETA_TEMP * (df["T2M"] - T_REF)))

# Prove leakage: regress Y_det on GHI + T2M → should be R²≈1.0
X_leak = sm.add_constant(df[["GHI", "T2M"]])
ols_leak = sm.OLS(df["Y_det"], X_leak).fit()

print(f"\n  OLS on deterministic target (Y = f(GHI, T2M)):")
print(f"  R²       = {ols_leak.rsquared:.6f}")
print(f"  Adj. R²  = {ols_leak.rsquared_adj:.6f}")
print(f"  RMSE     = {np.sqrt(ols_leak.mse_resid):.8f}")
print(f"  → R² ≈ 1.0 CONFIRMS ALGEBRAIC RECONSTRUCTION (Leakage proven)")

# Store leakage evidence
leakage_evidence = {
    "model": "OLS on deterministic target",
    "R2": ols_leak.rsquared,
    "adj_R2": ols_leak.rsquared_adj,
    "RMSE": np.sqrt(ols_leak.mse_resid),
    "n_obs": len(df),
    "interpretation": "Pseudo-perfect R2 from algebraic identity, not forecasting"
}

# ══════════════════════════════════════════════════════════════════════════
# PART B: STOCHASTIC TARGET RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: STOCHASTIC TARGET RECONSTRUCTION")
print("=" * 60)

n = len(df)

"""
Y_PV(t) = GHI(t) × η_ref × A × PR_stochastic(t) + ε_operational(t)

PR_stochastic(t) = PR_base
    × (1 - L_thermal(t))      # thermal degradation
    × (1 - L_cloud_resid(t))  # sub-GHI cloud intermittency
    × (1 - L_aerosol(t))      # aerosol attenuation
    × (1 - L_humidity(t))     # soiling + humidity
    × (1 - L_soiling(t))      # dust/peatland aerosol (Kalimantan-specific)

Each loss term is independently stochastic, parameterized from physics literature.
The stochastic target CANNOT be algebraically reconstructed from GHI + T2M alone.
"""

# ── LOSS COMPONENT 1: Thermal Loss ────────────────────────────────────────
# L_thermal = beta_pv * max(0, T2M - 25)
# For equatorial Bontang: T2M mean=26.92°C → thermal loss is real but small
# Add heteroskedastic noise: σ increases with temperature stress
T_excess         = np.maximum(0, df["T2M"].values - T_REF)
L_thermal_mean   = BETA_TEMP * T_excess
L_thermal_sigma  = 0.001 + 0.0005 * T_excess   # Heteroskedastic
L_thermal        = L_thermal_mean + np.random.normal(0, L_thermal_sigma, n)
L_thermal        = np.clip(L_thermal, 0.001, 0.12)

print(f"  L_thermal:   mean={L_thermal.mean():.4f}, std={L_thermal.std():.4f} "
      f"[range {L_thermal.min():.4f}–{L_thermal.max():.4f}]")

# ── LOSS COMPONENT 2: Cloud Residual Intermittency ────────────────────────
# GHI already reflects monthly-mean cloud attenuation.
# BUT within-month cloud intermittency (step changes, partial cloud events)
# creates additional stochastic loss NOT captured in monthly-mean GHI.
# Loss increases with cloud amount; variance is highest at intermediate cloud (60-80%)
CLOUD_norm       = df["CLOUD"].values / 100.0
L_cloud_mean     = 0.02 + 0.08 * CLOUD_norm           # 2–10% additional loss
L_cloud_sigma    = 0.015 + 0.04 * CLOUD_norm * (1 - CLOUD_norm)  # max variance at CLOUD=50%
L_cloud_resid    = L_cloud_mean + np.random.normal(0, L_cloud_sigma, n)
L_cloud_resid    = np.clip(L_cloud_resid, 0.005, 0.18)

print(f"  L_cloud_resid: mean={L_cloud_resid.mean():.4f}, std={L_cloud_resid.std():.4f}")

# ── LOSS COMPONENT 3: Aerosol Attenuation ────────────────────────────────
# Maritime aerosol: sea salt, sulfate baseline
# CRITICAL for Kalimantan: peatland fire aerosol (Sep-Oct 2015, 2019, 2023)
# Use Gamma distribution: positively skewed, physically realistic
# Fire season proxy: PRECTOT < 3 mm/day AND month ∈ {8,9,10} = elevated aerosol

fire_season = ((df["PRECTOT"].values < 3.0) &
               (df["MONTH"].isin([8, 9, 10]).values)).astype(float)

# Gamma(k, theta): mean = k*theta, var = k*theta²
# Baseline maritime aerosol: Gamma(2, 0.01) → mean=0.02
# Fire season: Gamma(3, 0.02) → mean=0.06
k_aerosol     = 2.0 + 1.5 * fire_season
theta_aerosol = 0.012 + 0.015 * fire_season
L_aerosol     = np.array([np.random.gamma(k, t) for k, t in zip(k_aerosol, theta_aerosol)])
L_aerosol     = np.clip(L_aerosol, 0.005, 0.15)

print(f"  L_aerosol:   mean={L_aerosol.mean():.4f}, std={L_aerosol.std():.4f} "
      f"[fire season months: {fire_season.sum():.0f}]")

# ── LOSS COMPONENT 4: Humidity / Soiling ──────────────────────────────────
# High tropical humidity → moisture film on panels → light scattering loss
# RH mean=84.6%, very high → persistent soiling contribution
RH_norm        = (df["RH"].values - 70) / 30.0  # normalize around 70-100%
RH_norm        = np.clip(RH_norm, 0, 1)
L_humidity_mean  = 0.005 + 0.025 * RH_norm
L_humidity_sigma = 0.006
L_humidity       = L_humidity_mean + np.random.normal(0, L_humidity_sigma, n)
L_humidity       = np.clip(L_humidity, 0.001, 0.06)

print(f"  L_humidity:  mean={L_humidity.mean():.4f}, std={L_humidity.std():.4f}")

# ── LOSS COMPONENT 5: Inverter / System Losses ────────────────────────────
# Beta distribution: bounded [0,1], flexible shape
# Beta(2, 15): mean ≈ 0.118 (11.8% system loss), realistic for tropical PV
alpha_inv, beta_inv = 2.0, 15.0
L_inverter = np.random.beta(alpha_inv, beta_inv, n)
L_inverter = np.clip(L_inverter, 0.04, 0.28)

print(f"  L_inverter:  mean={L_inverter.mean():.4f}, std={L_inverter.std():.4f}")

# ── LOSS COMPONENT 6: Monsoon / ITCZ Seasonal Variability (Q1 AUDIT FIX) ──
"""
Q1 CODE AUDIT FIX (2026-06-20): The manuscript (Table 2, Figure 6
caption) describes SEVEN stochastic loss components, including
L_monsoon ~ N(0, σ_seasonal) for ITCZ-related variability. This
component did not exist in the repository's committed code prior to
this fix — only 5 of the 7 documented components were implemented (see
39_CODE_AUDIT_CRITICAL_FINDINGS.md, finding #7). Added here.

Seasonal variance is largest during ITCZ transition months (Mar-Apr,
Sep-Oct for the equatorial Maritime Continent) and smallest during the
more stable mid-monsoon months, consistent with the manuscript's
physical motivation.
"""
itcz_transition = df["MONTH"].isin([3, 4, 9, 10]).values.astype(float)
sigma_seasonal   = 0.010 + 0.015 * itcz_transition   # higher variance in transition months
L_monsoon        = np.random.normal(0, sigma_seasonal, n)
L_monsoon        = np.clip(L_monsoon, -0.05, 0.05)

print(f"  L_monsoon:   mean={L_monsoon.mean():.4f}, std={L_monsoon.std():.4f} "
      f"[ITCZ transition months: {int(itcz_transition.sum())}]")

# ── LOSS COMPONENT 7: ENSO-Cloud Coupling (Q1 AUDIT FIX) ──────────────────
"""
Manuscript Table 2: L_ENSO ~ N(0, σ∝|ONI|) — ENSO-cloud coupling.

PIPELINE ORDERING NOTE: the ONI index proper is constructed in notebook
03 (03_feature_engineering.py), which runs AFTER this notebook in the
pipeline. To avoid introducing a circular cross-notebook dependency
(02 depending on 03's output, while 03's anomaly features are in turn
computed from data produced by 01/02), this component uses a
self-contained, simplified ENSO-amplitude PROXY: a smooth multi-year
oscillation with realistic period (~3-5 years, matching observed ENSO
recurrence) and amplitude, rather than importing the actual ONI series.
This proxy captures the INTENDED variance-amplification mechanism
(ENSO state modulates cloud-GHI coupling variance) without requiring
notebook 03 to run first. If exact consistency with the ONI series used
downstream in notebooks 03/06 is required, refactor this block to load
the same oni_noaa_cpc_real.csv (or its synthetic fallback) produced by
fetch_oni_dmi.py, called BEFORE this point in the pipeline.
"""
t_idx = np.arange(n)
enso_amplitude_proxy = 0.9 * np.sin(2 * np.pi * t_idx / 46) + 0.4 * np.sin(2 * np.pi * t_idx / 84)
sigma_enso = 0.008 * np.abs(enso_amplitude_proxy)
L_ENSO = np.random.normal(0, np.maximum(sigma_enso, 1e-4), n)
L_ENSO = np.clip(L_ENSO, -0.04, 0.04)

print(f"  L_ENSO:      mean={L_ENSO.mean():.4f}, std={L_ENSO.std():.4f} "
      f"[proxy amplitude range: {enso_amplitude_proxy.min():.2f} to {enso_amplitude_proxy.max():.2f}]")

# ── COMBINE INTO STOCHASTIC PERFORMANCE RATIO ────────────────────────────
PR_base        = 0.80   # Baseline PR for well-maintained tropical system
PR_stochastic  = (PR_base
                  * (1 - L_thermal)
                  * (1 - L_cloud_resid)
                  * (1 - L_aerosol)
                  * (1 - L_humidity)
                  * (1 - L_inverter)
                  * (1 - L_monsoon)
                  * (1 - L_ENSO))

print(f"\n  PR_stochastic: mean={PR_stochastic.mean():.4f}, std={PR_stochastic.std():.4f} "
      f"[range {PR_stochastic.min():.4f}–{PR_stochastic.max():.4f}]")

# ── OPERATIONAL NOISE (iid residual) ──────────────────────────────────────
# Captures: curtailment, temporary outages, measurement uncertainty
# Scale: ~2% of mean GHI output → σ_op ≈ 0.01
sigma_op     = 0.01 * df["GHI"].mean()
eps_op       = np.random.normal(0, sigma_op, n)

# ── FINAL STOCHASTIC PV TARGET ────────────────────────────────────────────
df["Y_stoch"] = (df["GHI"] * ETA_REF * A_PANEL * PR_stochastic + eps_op)
df["Y_stoch"] = np.maximum(df["Y_stoch"], 0)   # Non-negativity constraint

# Store individual loss components for decomposition analysis
df["L_thermal"]    = L_thermal
df["L_cloud_resid"]= L_cloud_resid
df["L_aerosol"]    = L_aerosol
df["L_humidity"]   = L_humidity
df["L_inverter"]   = L_inverter
df["L_monsoon"]    = L_monsoon     # Q1 audit fix: now stored (7th component)
df["L_ENSO"]       = L_ENSO        # Q1 audit fix: now stored (7th component)
df["PR_stoch"]     = PR_stochastic
df["fire_season"]  = fire_season

# ══════════════════════════════════════════════════════════════════════════
# PART C: LEAKAGE DEMONSTRATION (stochastic vs. deterministic)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C: LEAKAGE DEMONSTRATION — R² COMPARISON")
print("=" * 60)

# OLS on STOCHASTIC target with same predictors
X_sto = sm.add_constant(df[["GHI", "T2M"]])
ols_sto = sm.OLS(df["Y_stoch"], X_sto).fit()

print(f"\n  {'Model':<45} {'R²':>8} {'RMSE':>12}")
print(f"  {'-'*65}")
print(f"  {'OLS on DETERMINISTIC target (leakage proof)':<45} "
      f"{ols_leak.rsquared:>8.4f} {np.sqrt(ols_leak.mse_resid):>12.6f}")
print(f"  {'OLS on STOCHASTIC target (corrected)':<45} "
      f"{ols_sto.rsquared:>8.4f} {np.sqrt(ols_sto.mse_resid):>12.6f}")
print(f"\n  → R² DROP from deterministic to stochastic: "
      f"{ols_leak.rsquared - ols_sto.rsquared:.4f}")
print(f"  → This gap IS the deterministic leakage magnitude")
print(f"  → Stochastic R² is the scientifically valid benchmark")

# ══════════════════════════════════════════════════════════════════════════
# PART D: STOCHASTIC TARGET STATISTICS (for paper Table 1)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART D: STOCHASTIC TARGET DESCRIPTIVE STATISTICS")
print("=" * 60)

target_stats = pd.DataFrame({
    "Variable":   ["Y_stoch", "PR_stoch", "L_thermal", "L_cloud_resid",
                   "L_aerosol", "L_humidity", "L_inverter"],
    "Mean":  [df[c].mean()  for c in ["Y_stoch","PR_stoch","L_thermal","L_cloud_resid",
                                       "L_aerosol","L_humidity","L_inverter"]],
    "Std":   [df[c].std()   for c in ["Y_stoch","PR_stoch","L_thermal","L_cloud_resid",
                                       "L_aerosol","L_humidity","L_inverter"]],
    "Min":   [df[c].min()   for c in ["Y_stoch","PR_stoch","L_thermal","L_cloud_resid",
                                       "L_aerosol","L_humidity","L_inverter"]],
    "Max":   [df[c].max()   for c in ["Y_stoch","PR_stoch","L_thermal","L_cloud_resid",
                                       "L_aerosol","L_humidity","L_inverter"]],
}).round(5)

print(target_stats.to_string(index=False))

# Variance decomposition
Y_var    = df["Y_stoch"].var()
GHI_var  = (df["GHI"] * ETA_REF * PR_stochastic.mean()).var()
PR_var   = (df["GHI"].mean() * ETA_REF * PR_stochastic).var()

print(f"\n  Variance decomposition:")
print(f"  GHI variation explains:  {GHI_var/Y_var*100:.1f}% of Y_stoch variance")
print(f"  PR variation explains:   {PR_var/Y_var*100:.1f}% of Y_stoch variance")
print(f"  (sum > 100% due to covariance between GHI and PR components)")

# ── SAVE ──────────────────────────────────────────────────────────────────
df.to_parquet(f"{DATA_DIR}/02_target_reconstructed.parquet", index=False)
target_stats.to_csv(f"{OUT_DIR}/02_stochastic_target_stats.csv", index=False)

leak_df = pd.DataFrame([{
    "target_type": "deterministic", "R2": ols_leak.rsquared,
    "RMSE": np.sqrt(ols_leak.mse_resid)},
    {"target_type": "stochastic", "R2": ols_sto.rsquared,
    "RMSE": np.sqrt(ols_sto.mse_resid)}])
leak_df.to_csv(f"{OUT_DIR}/02_leakage_demonstration.csv", index=False)

print(f"\n✅ Notebook 02 complete.")
print(f"   Columns added: Y_stoch, PR_stoch, L_thermal, L_cloud_resid, L_aerosol, L_humidity, L_inverter, L_monsoon, L_ENSO")
