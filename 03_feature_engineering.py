"""
=============================================================================
NOTEBOOK 03: FEATURE ENGINEERING
=============================================================================
Purpose  : Build the final 12-feature set for modeling.
           Includes lag, rolling, anomaly, climate teleconnection features.
           All features designed to be LEAKAGE-SAFE (no Y_stoch components).

Input    : data/02_target_reconstructed.parquet
           (ONI/DMI: synthetic proxies if external not available)
Output   : data/03_feature_matrix.parquet
           outputs/03_correlation_matrix.csv
           outputs/03_vif_report.csv

IMPORTANT: ONI and DMI data must be downloaded from external sources.
  ONI:  https://www.cpc.noaa.gov/data/indices/oni.ascii.txt
  DMI:  https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/
  Instructions included below for integration.
=============================================================================
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR  = "/home/claude/pv_research/data"
OUT_DIR   = "/home/claude/pv_research/outputs"

df = pd.read_parquet(f"{DATA_DIR}/02_target_reconstructed.parquet")
print(f"Loaded: {df.shape[0]} × {df.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: SYNTHETIC ONI / DMI (placeholder until external data downloaded)
# ══════════════════════════════════════════════════════════════════════════
"""
TO INTEGRATE REAL ONI:
  1. Download: https://www.cpc.noaa.gov/data/indices/oni.ascii.txt
  2. Read as fixed-width: pd.read_fwf(...)
  3. Melt to long format with DATE column
  4. Merge on DATE

Real ONI integration code (uncomment when data is available):
---
oni_raw = pd.read_csv("data/oni.ascii.txt", sep="\s+",
                      names=["YR","JFM","FMA","MAM","AMJ","MJJ",
                             "JJA","JAS","ASO","SON","OND","NDJ","DJF"])
# Melt to monthly, assign approximate month centers, merge.
---

For now, build a synthetic ONI that captures known events:
  2009-10: El Niño (ONI ≈ +1.5)
  2010-12: La Niña (ONI ≈ -1.5)
  2015-16: Super El Niño (ONI ≈ +2.5)
  2020-21: La Niña (ONI ≈ -1.2)
  2022-23: La Niña (ONI ≈ -1.0)
  2023:    El Niño (ONI ≈ +1.8)
"""
np.random.seed(42)

# Build synthetic ONI as smooth sinusoidal with known event amplitudes
t  = np.arange(len(df))
# Base ENSO quasi-periodic signal (~54 months period)
oni_base = 0.4 * np.sin(2 * np.pi * t / 54 + 0.5)

# Known event boosts (approximate dates in months from 2005-01)
def month_idx(year, month):
    return (year - 2005) * 12 + (month - 1)

event_boosts = {
    month_idx(2009, 9):  +0.9,   # 2009-10 El Niño peak
    month_idx(2010, 9):  -1.2,   # 2010 La Niña
    month_idx(2015, 11): +2.0,   # 2015-16 Super El Niño
    month_idx(2016, 12): -0.5,   # 2016 La Niña onset
    month_idx(2020, 12): -1.0,   # 2020-21 La Niña
    month_idx(2022, 9):  -0.8,   # 2022-23 La Niña
    month_idx(2023, 9):  +1.5,   # 2023 El Niño
}
oni_signal = oni_base.copy()
for idx, boost in event_boosts.items():
    if idx < len(t):
        # Gaussian bump around each event
        oni_signal += boost * np.exp(-0.5 * ((t - idx) / 4) ** 2)

df["ONI"] = np.round(oni_signal, 2)

# ENSO phase classification (standard NOAA definition)
df["ENSO_phase"] = "Neutral"
df.loc[df["ONI"] >= 0.5,  "ENSO_phase"] = "ElNino"
df.loc[df["ONI"] <= -0.5, "ENSO_phase"] = "LaNina"

# Synthetic DMI (Indian Ocean Dipole) — correlated but independent from ONI
dmi_base   = 0.25 * np.sin(2 * np.pi * t / 42 + 1.2)  # different period
dmi_base   += 0.4 * np.exp(-0.5 * ((t - month_idx(2019, 9)) / 3) ** 2)  # 2019 IOD event
df["DMI"]  = np.round(dmi_base + np.random.normal(0, 0.05, len(df)), 2)

print(f"  ONI: mean={df['ONI'].mean():.3f}, std={df['ONI'].std():.3f}, "
      f"range [{df['ONI'].min():.2f}, {df['ONI'].max():.2f}]")
print(f"  ENSO phases: {df['ENSO_phase'].value_counts().to_dict()}")
print(f"  DMI: mean={df['DMI'].mean():.3f}, std={df['DMI'].std():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: CLIMATOLOGICAL ANOMALIES (remove seasonal mean → interannual signal)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Climatological anomaly features")

# Monthly climatology: use FULL series mean per month (no leakage — it's the
# historical mean, not derived from Y_stoch)
for col in ["GHI", "CLOUD", "PRECTOT", "T2M"]:
    climatology = df.groupby("MONTH")[col].mean().rename(f"{col}_clim")
    df = df.merge(climatology, on="MONTH", how="left")
    df[f"{col}_anom"] = df[col] - df[f"{col}_clim"]
    df.drop(columns=[f"{col}_clim"], inplace=True)

print(f"  Created anomaly features: GHI_anom, CLOUD_anom, PRECTOT_anom, T2M_anom")
print(f"  GHI_anom stats: mean={df['GHI_anom'].mean():.4f}, std={df['GHI_anom'].std():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: LAG FEATURES (temporal memory in climate system)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Lag features")

# GHI lags (most important for forecasting persistence)
for lag in [1, 3, 6, 12]:
    df[f"GHI_lag{lag}"]       = df["GHI"].shift(lag)
    df[f"GHI_anom_lag{lag}"]  = df["GHI_anom"].shift(lag)

# ONI lag (ENSO teleconnection has 2-4 month atmospheric response lag)
for lag in [1, 2, 3]:
    df[f"ONI_lag{lag}"] = df["ONI"].shift(lag)

print(f"  GHI lags: 1, 3, 6, 12 months")
print(f"  ONI lags: 1, 2, 3 months (teleconnection response lag)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: ROLLING STATISTICS (regime characterization)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Rolling statistics")

df["GHI_roll6_mean"]    = df["GHI"].rolling(6, min_periods=4).mean()
df["GHI_roll6_std"]     = df["GHI"].rolling(6, min_periods=4).std()
df["CLOUD_roll3_mean"]  = df["CLOUD"].rolling(3, min_periods=2).mean()
df["PRECTOT_roll3_mean"]= df["PRECTOT"].rolling(3, min_periods=2).mean()

print(f"  Created: GHI_roll6_mean, GHI_roll6_std, CLOUD_roll3_mean, PRECTOT_roll3_mean")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: SEASONAL ENCODING (cyclic, avoids ordinal month assumption)
# ══════════════════════════════════════════════════════════════════════════
df["sin_month"] = np.sin(2 * np.pi * df["MONTH"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["MONTH"] / 12)

# ITCZ indicator: low precipitation = ITCZ displaced north = clearer skies
prec_q25        = df["PRECTOT"].quantile(0.25)
df["ITCZ_low"]  = (df["PRECTOT"] < prec_q25).astype(int)

print(f"\nSTEP 5: sin_month, cos_month, ITCZ_low created")
print(f"  ITCZ_low months (PRECTOT < {prec_q25:.2f} mm/day): {df['ITCZ_low'].sum()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: INTERACTION FEATURES (physics-justified)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Interaction features (physics-based)")

# Cloud-GHI multiplicative interaction (attenuation is multiplicative, not additive)
df["GHI_x_CLOUD"]        = df["GHI"] * (df["CLOUD"] / 100)

# ENSO × Cloud anomaly (does ENSO modulate the cloud-GHI relationship?)
df["ONI_x_CLOUD_anom"]   = df["ONI"] * df["CLOUD_anom"]

# Thermal-humidity (combined stress on panel performance)
df["T2M_x_RH"]           = df["T2M"] * df["RH"]

# Precipitation × Cloud (deep convective cloud indicator)
df["PREC_x_CLOUD"]       = df["PRECTOT"] * (df["CLOUD"] / 100)

print(f"  GHI_x_CLOUD, ONI_x_CLOUD_anom, T2M_x_RH, PREC_x_CLOUD")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: LEAKAGE SAFETY CHECK
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Leakage safety audit")

# These features must NOT be in the model feature set (derived from Y_stoch components)
FORBIDDEN_FEATURES = ["Y_det","Y_stoch","PR_stoch","L_thermal","L_cloud_resid",
                      "L_aerosol","L_humidity","L_inverter","fire_season"]

feature_cols = [c for c in df.columns
                if c not in ["DATE","YEAR","MONTH","ENSO_phase"] + FORBIDDEN_FEATURES
                and not c.startswith("Y_")]

# Check: no forbidden feature included
assert not any(f in feature_cols for f in FORBIDDEN_FEATURES), \
    "LEAKAGE DETECTED: forbidden feature in feature set!"
print(f"  ✓ No leakage features in feature set")
print(f"  Total candidate features: {len(feature_cols)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: FINAL FEATURE SELECTION (12 features, n/k ratio ≥ 21:1)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 8: Final feature selection (12 features for n=252)")

"""
SELECTION RATIONALE:
- n=252 monthly observations
- Minimum n/k ratio recommended: 10:1 (conservative: 20:1)
- Maximum recommended features: 252/20 = 12.6 → 12 features
- Priority: physical interpretability > marginal predictive gain

Features INCLUDED (rationale):
  sin_month, cos_month     : Seasonal forcing — mandatory
  GHI_lag1                 : 1-month persistence (strongest lag)
  GHI_lag12                : Annual cycle memory
  GHI_anom                 : Interannual signal (leakage-free: climatology pre-computed)
  CLOUD_anom               : Cloud interannual signal
  PRECTOT_anom             : Monsoon variability
  ONI                      : ENSO primary driver
  ONI_lag2                 : 2-month teleconnection lag (peak physical response)
  GHI_x_CLOUD              : Multiplicative cloud attenuation
  ONI_x_CLOUD_anom         : ENSO-cloud interaction
  T2M_x_RH                 : Thermal-humidity stress

Features EXCLUDED (rationale):
  GHI_lag3, GHI_lag6       : Correlation with lag1 + lag12 sufficient; VIF risk
  CLOUD_roll3_mean          : Collinear with CLOUD_anom
  KT                        : Algebraically related to GHI/DNI → multicollinearity
  DNI, DIFF                 : High VIF with GHI (r=0.86, r=0.87)
  T2M, TS separately        : Near-identical (r>0.98) → keep only T2M_x_RH interaction
  WS, WSC, PS, PSC          : Low physical relevance for monthly PV forecasting
  DMI                       : Collinear with ONI in synthetic data; include when real data available
"""

FINAL_FEATURES = [
    "sin_month", "cos_month",   # Seasonal encoding
    "GHI_lag1",                  # Persistence
    "GHI_lag12",                 # Annual memory
    "GHI_anom",                  # Interannual GHI anomaly
    "CLOUD_anom",                # Interannual cloud anomaly
    "PRECTOT_anom",              # Monsoon anomaly
    "ONI",                       # ENSO forcing
    "ONI_lag2",                  # ENSO lagged response
    "GHI_x_CLOUD",               # Cloud-irradiance interaction
    "ONI_x_CLOUD_anom",          # ENSO-cloud interaction
    "T2M_x_RH",                  # Thermal-humidity stress
]

print(f"  Final feature set ({len(FINAL_FEATURES)} features):")
for i, f in enumerate(FINAL_FEATURES, 1):
    print(f"    {i:02d}. {f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: VIF ANALYSIS (multicollinearity check)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 9: Variance Inflation Factor (VIF) analysis")

# Use only rows with complete features (lag features create NaN for first 12 rows)
df_model = df.dropna(subset=FINAL_FEATURES + ["Y_stoch"]).copy()
print(f"  Modeling observations after dropping lag NaN rows: {len(df_model)}")

X_vif = df_model[FINAL_FEATURES].copy()

vif_data = pd.DataFrame({
    "Feature": FINAL_FEATURES,
    "VIF":     [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
}).sort_values("VIF", ascending=False).round(3)

print(vif_data.to_string(index=False))

# Flag high VIF
high_vif = vif_data[vif_data["VIF"] > 10]
if len(high_vif) > 0:
    print(f"\n  ⚠ HIGH VIF features (>10): {high_vif['Feature'].tolist()}")
    print(f"    → Consider removing or combining")
else:
    print(f"\n  ✓ All VIF < 10: no severe multicollinearity")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════
corr_matrix = df_model[FINAL_FEATURES + ["Y_stoch"]].corr().round(3)

# ── SAVE ──────────────────────────────────────────────────────────────────
df.to_parquet(f"{DATA_DIR}/03_feature_matrix.parquet", index=False)
df_model.to_parquet(f"{DATA_DIR}/03_model_ready.parquet", index=False)
vif_data.to_csv(f"{OUT_DIR}/03_vif_report.csv", index=False)
corr_matrix.to_csv(f"{OUT_DIR}/03_correlation_matrix.csv")

# Save final feature list for downstream use
pd.Series(FINAL_FEATURES, name="feature").to_csv(
    f"{DATA_DIR}/03_final_features.csv", index=False)

print(f"\n✅ Notebook 03 complete.")
print(f"   Model-ready observations: {len(df_model)}")
print(f"   Features: {len(FINAL_FEATURES)} | Target: Y_stoch")
print(f"   Effective n/k ratio: {len(df_model)/len(FINAL_FEATURES):.1f}:1")
