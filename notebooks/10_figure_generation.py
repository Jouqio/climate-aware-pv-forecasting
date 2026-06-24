"""
=============================================================================
NOTEBOOK 10: FIGURE GENERATION
=============================================================================
Purpose  : Generate all publication-ready figures for Q1/Q2 submission.
           13 figures targeting Applied Energy / ECM journal standards. 
           All figures: 300 DPI, vector-ready, consistent styling.

Input    : All outputs from notebooks 01–09
Output   : figures/fig01_research_framework.png
           figures/fig02_leakage_demonstration.png
           figures/fig03_data_profile.png
           figures/fig04_seasonal_climatology.png
           figures/fig05_enso_teleconnection.png
           figures/fig06_stochastic_target_architecture.png
           figures/fig07_walkforward_scheme.png
           figures/fig08_model_performance.png
           figures/fig09_probabilistic_forecast.png
           figures/fig10_shap_summary.png
           figures/fig11_ols_shap_correspondence.png
           figures/fig12_residual_diagnostics.png
           figures/fig13_enso_phase_forecasting.png

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR / "outputs"
FIG_DIR    = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

COLORS = {
    "ols":     "#2166AC",   # Blue
    "sarimax": "#1B7837",   # Green
    "xgb":     "#D6604D",   # Red-orange
    "clim":    "#888888",   # Gray
    "enso_en": "#D73027",   # El Niño red
    "enso_ln": "#4575B4",   # La Niña blue
    "enso_nu": "#74ADD1",   # Neutral light blue
    "shap_pos":"#D73027",
    "shap_neg":"#4575B4",
}

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")

# Q1 AUDIT FIX: GHI_anom/CLOUD_anom/PRECTOT_anom/T2M_anom are no longer
# precomputed in 03_model_ready.parquet (they are now computed per-fold
# downstream to eliminate the leakage documented in
# 39_CODE_AUDIT_CRITICAL_FINDINGS.md, finding #1). For this notebook's
# purpose -- static, descriptive visualisation of the full 21-year
# series, not predictive evaluation -- recomputing them on the full
# sample here is methodologically legitimate (identical to the
# df_report pattern used in notebooks 03/05/07/08 for VIF/SHAP
# diagnostics). These columns must NEVER be fed back into any walk-
# forward model fit.
RAW_ANOMALY_SOURCE_COLS = ["GHI", "CLOUD", "PRECTOT", "T2M"]
for _col in RAW_ANOMALY_SOURCE_COLS:
    if f"{_col}_anom" not in df.columns:
        df[f"{_col}_anom"] = df[_col] - df.groupby("MONTH")[_col].transform("mean")
if "ONI_x_CLOUD_anom" not in df.columns and "ONI" in df.columns:
    df["ONI_x_CLOUD_anom"] = df["ONI"] * df["CLOUD_anom"]

# Load predictions (with fallback to synthetic if pipeline not fully run)
try:
    df_ols = pd.read_parquet(f"{DATA_DIR}/05_ols_predictions.parquet")
    df_sar = pd.read_parquet(f"{DATA_DIR}/06_sarimax_predictions.parquet")
    df_xgb = pd.read_parquet(f"{DATA_DIR}/07_xgboost_predictions.parquet")
    predictions_available = True
except FileNotFoundError:
    predictions_available = False
    print("  ⚠ Prediction files not found — generating figures with available data only")

try:
    shap_summary = pd.read_csv(f"{OUT_DIR}/08_shap_feature_summary.csv")
    xai_corr     = pd.read_csv(f"{OUT_DIR}/08_econometric_xai_correspondence.csv")
    ols_coef     = pd.read_csv(f"{OUT_DIR}/05_ols_coefficients.csv")
    shap_available = True
except FileNotFoundError:
    shap_available = False

print(f"Data loaded: {len(df)} obs | Predictions: {predictions_available} | SHAP: {shap_available}")

# ══════════════════════════════════════════════════════════════════════════
# FIG 01: RESEARCH FRAMEWORK DIAGRAM
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 01: Research Framework...")

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
ax.set_facecolor("white")

def draw_box(ax, x, y, w, h, text, color, fontsize=9, alpha=0.85):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="white",
                          linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4, wrap=True,
            multialignment="center")

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555555",
                                lw=1.5, connectionstyle="arc3,rad=0"))

# Layer 1: Data
draw_box(ax, 2,   6.2, 2.8, 0.8, "NASA POWER\n2005–2025\n(252 observations)", "#2166AC")
draw_box(ax, 5.5, 6.2, 2.8, 0.8, "ONI / DMI / SOI\nClimate Indices\n(NOAA CPC)", "#4393C3")
draw_box(ax, 9.0, 6.2, 2.8, 0.8, "Stochastic Target\nReconstruction\n(7 loss components)", "#D6604D")

# Layer 2: Features
draw_box(ax, 3.5, 4.7, 5.5, 0.8, "Feature Engineering\n(12 features: lags, anomalies, interactions, seasonal encoding)", "#35978F")

# Layer 3: Diagnostics
draw_box(ax, 1.5, 3.2, 2.5, 0.8, "Econometric\nDiagnostics\n(ADF, BG, White, VIF)", "#762A83")
draw_box(ax, 4.5, 3.2, 2.5, 0.8, "Walk-Forward\nValidation\n(9 expanding folds)", "#1B7837")
draw_box(ax, 7.5, 3.2, 2.5, 0.8, "Structural Break\nAnalysis\n(Bai-Perron, Chow)", "#8C510A")

# Layer 4: Models
draw_box(ax, 2.0, 1.7, 2.2, 0.8, "OLS-HC3\nEconometric", "#2166AC")
draw_box(ax, 4.5, 1.7, 2.2, 0.8, "SARIMAX+ONI\nClimate-Aware", "#1B7837")
draw_box(ax, 7.0, 1.7, 2.2, 0.8, "XGBoost\n(Constrained)", "#D6604D")
draw_box(ax, 9.5, 1.7, 2.2, 0.8, "SHAP / XAI\nCorrespondence", "#762A83")

# Layer 5: Output
draw_box(ax, 5.75, 0.4, 8.5, 0.8,
         "Probabilistic Forecasts + DM Test + Friedman Ranking + ENSO-Conditioned Analysis → Q1/Q2 Paper",
         "#543005", fontsize=9)

# Arrows
for x in [2, 5.5, 9]:
    draw_arrow(ax, x, 5.8, x if x != 5.5 else 3.5, 5.1)
draw_arrow(ax, 5.5, 5.8, 3.5, 5.1)
for x in [1.5, 4.5, 7.5]:
    draw_arrow(ax, 3.5, 4.3, x, 3.6)
for x in [2, 4.5, 7, 9.5]:
    draw_arrow(ax, 4.5 if x<6 else 6 if x<8 else 7.5, 2.8, x, 2.1)
for x in [2, 4.5, 7, 9.5]:
    draw_arrow(ax, x, 1.3, 5.75, 0.8)

ax.set_title("Research Framework: Climate-Aware Stochastic PV Forecasting for the Maritime Continent",
             fontsize=12, fontweight="bold", pad=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig01_research_framework.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  ✓ Fig 01 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 02: DETERMINISTIC LEAKAGE DEMONSTRATION
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 02: Deterministic Leakage...")

ETA_REF = 0.18; BETA_TEMP = 0.004; T_REF = 25.0
Y_det = ETA_REF * df["GHI"] * (1 - BETA_TEMP * (df["T2M"] - T_REF))
Y_sto = df["Y_stoch"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: Deterministic target vs GHI
axes[0].scatter(df["GHI"], Y_det, s=15, alpha=0.5, color=COLORS["ols"])
m, b = np.polyfit(df["GHI"], Y_det, 1)
x_line = np.linspace(df["GHI"].min(), df["GHI"].max(), 100)
axes[0].plot(x_line, m*x_line+b, "r-", lw=2)
r2_det = np.corrcoef(df["GHI"], Y_det)[0,1]**2
axes[0].set_title(f"(a) Deterministic Target\nR² with GHI = {r2_det:.4f} (pseudo-perfect leakage)")
axes[0].set_xlabel("GHI (kWh/m²/day)"); axes[0].set_ylabel("Y_PV_det (kWh/m²/day)")
axes[0].text(0.05, 0.93, f"R²={r2_det:.4f}", transform=axes[0].transAxes,
             fontsize=10, fontweight="bold", color="red",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Panel B: Stochastic target vs GHI
axes[1].scatter(df["GHI"], Y_sto, s=15, alpha=0.5, color=COLORS["xgb"])
m2, b2 = np.polyfit(df["GHI"], Y_sto, 1)
axes[1].plot(x_line, m2*x_line+b2, "b-", lw=2)
r2_sto = np.corrcoef(df["GHI"], Y_sto)[0,1]**2
axes[1].set_title(f"(b) Stochastic Target\nR² with GHI = {r2_sto:.4f} (realistic)")
axes[1].set_xlabel("GHI (kWh/m²/day)"); axes[1].set_ylabel("Y_PV_stoch (kWh/m²/day)")
axes[1].text(0.05, 0.93, f"R²={r2_sto:.4f}", transform=axes[1].transAxes,
             fontsize=10, fontweight="bold", color="blue",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Panel C: R² comparison bar chart
categories = ["Deterministic\nTarget", "Stochastic\nTarget\n(Corrected)"]
r2_vals    = [r2_det, r2_sto]
bar_colors = [COLORS["xgb"], COLORS["ols"]]
bars = axes[2].bar(categories, r2_vals, color=bar_colors, alpha=0.85, width=0.5, edgecolor="white")
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel("R² with GHI")
axes[2].set_title("(c) Leakage Magnitude\nR² Drop = Leakage Correction")
for bar, val in zip(bars, r2_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.4f}",
                 ha="center", va="bottom", fontweight="bold")
axes[2].axhline(0.95, color="red", ls="--", lw=1.5, alpha=0.7, label="Pseudo-R²≥0.95")
axes[2].legend()

fig.suptitle("Deterministic Leakage Demonstration: Pseudo-Perfect R² vs Stochastic Correction",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig02_leakage_demonstration.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  ✓ Fig 02 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 03: 21-YEAR DATA PROFILE
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 03: Data Profile...")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
dates = df["DATE"]

# Panel A: GHI with annual mean
axes[0].plot(dates, df["GHI"], color=COLORS["ols"], lw=1.2, alpha=0.8, label="GHI monthly")
annual_ghi = df.groupby("YEAR")["GHI"].mean()
for yr, val in annual_ghi.items():
    axes[0].axhline(val, xmin=((yr-2005)/21), xmax=((yr-2004)/21),
                    color="orange", lw=2.5, alpha=0.6)
axes[0].set_ylabel("GHI\n(kWh/m²/day)"); axes[0].legend(loc="upper right")
axes[0].set_title("(a) Global Horizontal Irradiance (GHI)")

# Panel B: CLOUD with ENSO phase shading
axes[1].fill_between(dates, df["CLOUD"], alpha=0.4, color=COLORS["sarimax"])
axes[1].plot(dates, df["CLOUD"], color=COLORS["sarimax"], lw=0.8)
# Shade El Niño periods
en_mask = df["ENSO_phase"] == "ElNino"
axes[1].fill_between(dates, 0, 100, where=en_mask, alpha=0.15,
                     color=COLORS["enso_en"], label="El Niño period")
axes[1].set_ylabel("Cloud Amount\n(%)"); axes[1].legend(loc="upper right")
axes[1].set_title("(b) Cloud Amount with El Niño periods shaded")
axes[1].set_ylim(40, 105)

# Panel C: Precipitation
axes[2].fill_between(dates, df["PRECTOT"], alpha=0.5, color="#762A83")
axes[2].set_ylabel("Precipitation\n(mm/day)"); axes[2].set_title("(c) Precipitation (IMERG)")

# Panel D: ONI
axes[3].fill_between(dates, df["ONI"], where=df["ONI"] > 0,
                     color=COLORS["enso_en"], alpha=0.7, label="El Niño (ONI>0)")
axes[3].fill_between(dates, df["ONI"], where=df["ONI"] < 0,
                     color=COLORS["enso_ln"], alpha=0.7, label="La Niña (ONI<0)")
axes[3].axhline(0.5,  color=COLORS["enso_en"], lw=1.5, ls="--", alpha=0.6)
axes[3].axhline(-0.5, color=COLORS["enso_ln"], lw=1.5, ls="--", alpha=0.6)
axes[3].axhline(0, color="black", lw=0.8)
axes[3].set_ylabel("ONI (°C)"); axes[3].legend(loc="upper right")
axes[3].set_title("(d) Oceanic Niño Index (ONI) — ENSO driver")

for ax in axes:
    ax.set_xlim(dates.min(), dates.max())
axes[-1].set_xlabel("Year")
fig.suptitle("NASA POWER Dataset Profile: Bontang, Kalimantan (2005–2025)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig03_data_profile.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ Fig 03 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 04: SEASONAL CLIMATOLOGY
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 04: Seasonal Climatology...")

month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Monthly box plots — GHI
ghi_by_month = [df[df["MONTH"]==m]["GHI"].values for m in range(1, 13)]
bp1 = axes[0].boxplot(ghi_by_month, patch_artist=True,
                       medianprops=dict(color="red", lw=2))
for patch in bp1["boxes"]:
    patch.set_facecolor(COLORS["ols"]); patch.set_alpha(0.7)
axes[0].set_xticklabels(month_labels)
axes[0].set_xlabel("Month"); axes[0].set_ylabel("GHI (kWh/m²/day)")
axes[0].set_title("(a) Monthly GHI Distribution (2005–2025)")

# Monthly box plots — Cloud Amount
cloud_by_month = [df[df["MONTH"]==m]["CLOUD"].values for m in range(1, 13)]
bp2 = axes[1].boxplot(cloud_by_month, patch_artist=True,
                       medianprops=dict(color="white", lw=2))
for patch in bp2["boxes"]:
    patch.set_facecolor(COLORS["sarimax"]); patch.set_alpha(0.7)
axes[1].set_xticklabels(month_labels)
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Cloud Amount (%)")
axes[1].set_title("(b) Monthly Cloud Amount Distribution")

fig.suptitle("Seasonal Climatology: Bontang Equatorial Maritime Pattern",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig04_seasonal_climatology.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ Fig 04 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 05: ENSO TELECONNECTION
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 05: ENSO Teleconnection...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: GHI anomaly vs ONI scatter by ENSO phase
colors_phase = {"ElNino": COLORS["enso_en"], "Neutral": COLORS["enso_nu"], "LaNina": COLORS["enso_ln"]}
for phase, grp in df.groupby("ENSO_phase"):
    axes[0].scatter(grp["ONI"], grp["GHI_anom"], s=20, alpha=0.7,
                    color=colors_phase.get(phase, "gray"), label=phase)
axes[0].axhline(0, color="black", lw=1, ls="--")
axes[0].axvline(0, color="black", lw=1, ls="--")
m_oni, b_oni = np.polyfit(df["ONI"], df["GHI_anom"], 1)
x_oni = np.linspace(df["ONI"].min(), df["ONI"].max(), 100)
axes[0].plot(x_oni, m_oni*x_oni+b_oni, "k-", lw=2, alpha=0.8)
r_oni = np.corrcoef(df["ONI"], df["GHI_anom"])[0,1]
axes[0].set_xlabel("ONI (°C)"); axes[0].set_ylabel("GHI Anomaly (kWh/m²/day)")
axes[0].set_title(f"(a) ONI vs GHI Anomaly\nr = {r_oni:.3f}")
axes[0].legend(fontsize=8)

# Panel B: Cross-correlation ONI(t-k) vs GHI_anom(t)
max_lag = 12
lags    = range(-max_lag, max_lag+1)
xcorr   = [np.corrcoef(df["ONI"].shift(lag).fillna(0), df["GHI_anom"])[0,1]
           for lag in lags]
axes[1].bar(lags, xcorr, color=[COLORS["enso_en"] if x>0 else COLORS["enso_ln"] for x in xcorr],
            alpha=0.75, width=0.8)
axes[1].axhline(0, color="black", lw=1)
axes[1].axhline(1.96/np.sqrt(len(df)), color="gray", ls="--", lw=1.2, label="±95% CI")
axes[1].axhline(-1.96/np.sqrt(len(df)), color="gray", ls="--", lw=1.2)
peak_lag = list(lags)[np.argmax(np.abs(xcorr))]
axes[1].axvline(peak_lag, color="red", lw=1.5, ls="-.", alpha=0.7, label=f"Peak lag={peak_lag}mo")
axes[1].set_xlabel("Lag (months)"); axes[1].set_ylabel("Cross-correlation")
axes[1].set_title("(b) Cross-correlation: ONI lag vs GHI Anomaly")
axes[1].legend()

# Panel C: ENSO phase-conditioned GHI distribution
enso_ghi = {p: df[df["ENSO_phase"]==p]["GHI"].values for p in ["ElNino","Neutral","LaNina"]}
bp = axes[2].boxplot([enso_ghi.get(p, [0]) for p in ["ElNino","Neutral","LaNina"]],
                      patch_artist=True, medianprops=dict(color="white", lw=2.5))
for patch, color in zip(bp["boxes"],
                        [COLORS["enso_en"], COLORS["enso_nu"], COLORS["enso_ln"]]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
axes[2].set_xticklabels(["El Niño", "Neutral", "La Niña"])
axes[2].set_ylabel("GHI (kWh/m²/day)")
axes[2].set_title("(c) GHI Distribution by ENSO Phase")

fig.suptitle("ENSO Teleconnection Analysis: ONI–GHI Coupling in the Maritime Continent",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig05_enso_teleconnection.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ Fig 05 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 06: STOCHASTIC TARGET ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 06: Stochastic Target Architecture...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Loss component distributions
loss_cols = ["L_thermal","L_cloud_resid","L_aerosol","L_humidity","L_inverter"]
loss_labels = ["Thermal\nLoss","Cloud\nIntermittency","Aerosol\nAttenuation",
               "Humidity\nSoiling","Inverter\nLoss"]
loss_colors = ["#D73027","#FC8D59","#FEE090","#91BFDB","#4575B4"]

for i, (col, label, color) in enumerate(zip(loss_cols, loss_labels, loss_colors)):
    if col in df.columns:
        val = df[col].values
        axes[0].bar(i, val.mean(), color=color, alpha=0.85, width=0.6,
                    edgecolor="white", linewidth=1.5,
                    yerr=val.std(), capsize=5, error_kw={"linewidth":2})
        axes[0].text(i, val.mean()/2, f"{val.mean()*100:.1f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white")

axes[0].set_xticks(range(len(loss_cols)))
axes[0].set_xticklabels(loss_labels, fontsize=8)
axes[0].set_ylabel("Mean Loss Fraction")
axes[0].set_title("(a) Stochastic Loss Components\n(mean ± std, 252 months)")

# Panel B: Performance Ratio distribution by ENSO phase
if "PR_stoch" in df.columns:
    for phase, color in [("ElNino", COLORS["enso_en"]),
                         ("Neutral", COLORS["enso_nu"]),
                         ("LaNina", COLORS["enso_ln"])]:
        mask = df["ENSO_phase"] == phase
        if mask.sum() > 5:
            pr_vals = df.loc[mask, "PR_stoch"].values
            axes[1].hist(pr_vals, bins=20, alpha=0.5, color=color,
                         label=f"{phase} (n={mask.sum()})", density=True)

axes[1].set_xlabel("Performance Ratio (PR_stoch)")
axes[1].set_ylabel("Density")
axes[1].set_title("(b) Performance Ratio Distribution\nby ENSO Phase")
axes[1].legend()

fig.suptitle("Stochastic PV Target Architecture: Physics-Based Uncertainty Decomposition",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig06_stochastic_target_architecture.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ Fig 06 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 07: WALK-FORWARD VALIDATION SCHEME
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 07: Walk-Forward Validation Scheme...")

fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(2005, 2026); ax.set_ylim(0, 12)
ax.set_xlabel("Year"); ax.set_title(
    "Walk-Forward Expanding Window Validation Scheme\n(9 folds + final holdout)",
    fontsize=11, fontweight="bold")
ax.axis("off")

y_labels = []
for fold_idx, test_year in enumerate(range(2015, 2025)):
    y = 10.5 - fold_idx
    # Training bar (green)
    ax.barh(y, test_year - 2005, left=2005, height=0.6,
            color=COLORS["sarimax"], alpha=0.7)
    # Test bar (orange or red for holdout)
    bar_color = "#D6604D" if test_year <= 2023 else "#8C510A"
    ax.barh(y, 1, left=test_year, height=0.6, color=bar_color, alpha=0.9)
    label = f"Fold {fold_idx+1}" if test_year <= 2023 else "Holdout"
    y_labels.append((y, f"{label}: Train {2005}–{test_year-1} | Test {test_year}"))

ax.set_yticks([r[0] for r in y_labels])
ax.set_yticklabels([r[1] for r in y_labels], fontsize=9)
ax.set_xticks(range(2005, 2026))
ax.set_xticklabels(range(2005, 2026), rotation=45, ha="right", fontsize=8)

legend_els = [
    mpatches.Patch(color=COLORS["sarimax"], alpha=0.7, label="Training set (expanding)"),
    mpatches.Patch(color="#D6604D", alpha=0.9, label="Test set (walk-forward)"),
    mpatches.Patch(color="#8C510A", alpha=0.9, label="Final holdout (unused in CV)"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=9)
ax.axvline(2015, color="gray", lw=1.5, ls="--", alpha=0.5)
ax.axvline(2024, color="#8C510A", lw=2, ls="-", alpha=0.7)
ax.axis("on")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig07_walkforward_scheme.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ Fig 07 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 12: RESIDUAL DIAGNOSTICS (always available — uses OLS residuals)
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 12: Residual Diagnostics...")

import statsmodels.api as sm
X_full = sm.add_constant(df[pd.read_csv(f"{DATA_DIR}/03_final_features.csv")["feature"].tolist()])
ols_full = sm.OLS(df["Y_stoch"], X_full).fit()
resid = ols_full.resid.values

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(resid, lags=24, ax=axes[0,0], title="(a) ACF of OLS Residuals")

# Residual vs Fitted
axes[0,1].scatter(ols_full.fittedvalues, resid, s=15, alpha=0.5, color=COLORS["ols"])
axes[0,1].axhline(0, color="red", lw=2)
axes[0,1].set_xlabel("Fitted Values"); axes[0,1].set_ylabel("Residuals")
axes[0,1].set_title("(b) Residuals vs. Fitted")

# Q-Q plot
stats.probplot(resid, dist="norm", plot=axes[1,0])
axes[1,0].set_title("(c) Normal Q-Q Plot of Residuals")
axes[1,0].get_lines()[0].set(markersize=4, alpha=0.5)

# Residual time series colored by ENSO phase
dates = df["DATE"].values
phase_colors = df["ENSO_phase"].map({"ElNino": COLORS["enso_en"],
                                     "Neutral": COLORS["enso_nu"],
                                     "LaNina": COLORS["enso_ln"]}).values
axes[1,1].scatter(dates, resid, c=phase_colors, s=15, alpha=0.7)
axes[1,1].axhline(0, color="black", lw=1.5)
axes[1,1].set_xlabel("Date"); axes[1,1].set_ylabel("Residual")
axes[1,1].set_title("(d) Residual Time Series by ENSO Phase")
for phase, color in [("El Niño", COLORS["enso_en"]),
                     ("Neutral", COLORS["enso_nu"]),
                     ("La Niña", COLORS["enso_ln"])]:
    axes[1,1].scatter([], [], c=color, s=30, label=phase)
axes[1,1].legend()

fig.suptitle("OLS-HC3 Residual Diagnostics", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig12_residual_diagnostics.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ Fig 12 saved")

# ══════════════════════════════════════════════════════════════════════════
# FIG 08: MODEL PERFORMANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("Generating Fig 08: Model Performance Comparison...")

if predictions_available:
    df_ols = pd.read_parquet(f"{DATA_DIR}/05_ols_predictions.parquet")
    df_sar = pd.read_parquet(f"{DATA_DIR}/06_sarimax_predictions.parquet")
    df_xgb = pd.read_parquet(f"{DATA_DIR}/07_xgboost_predictions.parquet")
    df_perf = df_ols[['DATE','y_true','y_pred_ols']].merge(
        df_sar[['DATE','y_pred_sarimax']], on='DATE')
    df_perf = df_perf.merge(df_xgb[['DATE','y_pred_xgb']], on='DATE')

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df_perf['DATE'], df_perf['y_true'], color='black', lw=2, label='Actual')
    axes[0].plot(df_perf['DATE'], df_perf['y_pred_ols'], color=COLORS['ols'], lw=1.5, label='OLS-HC3')
    axes[0].plot(df_perf['DATE'], df_perf['y_pred_sarimax'], color=COLORS['sarimax'], lw=1.5, label='SARIMAX+ONI')
    axes[0].plot(df_perf['DATE'], df_perf['y_pred_xgb'], color=COLORS['xgb'], lw=1.5, label='XGBoost')
    axes[0].set_ylabel('Y_stoch'); axes[0].set_title('(a) Actual vs Model Predictions')
    axes[0].legend(loc='upper right')

    rmse_ols = np.sqrt(np.mean((df_perf['y_true'] - df_perf['y_pred_ols'])**2))
    rmse_sar = np.sqrt(np.mean((df_perf['y_true'] - df_perf['y_pred_sarimax'])**2))
    rmse_xgb = np.sqrt(np.mean((df_perf['y_true'] - df_perf['y_pred_xgb'])**2))
    x_pos = np.arange(3)
    rmse_values = [rmse_ols, rmse_sar, rmse_xgb]
    axes[1].bar(x_pos, rmse_values,
                color=[COLORS['ols'], COLORS['sarimax'], COLORS['xgb']], alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['OLS-HC3','SARIMAX+ONI','XGBoost'])
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
    axes[1].set_ylabel('RMSE'); axes[1].set_title('(b) Walk-forward RMSE by Model')
    fig.suptitle('Model Performance Comparison: Actual vs Predictions and RMSE', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig08_model_performance.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ✓ Fig 08 saved')
else:
    print('  ⚠ Fig 08 skipped: prediction files unavailable')

# ══════════════════════════════════════════════════════════════════════════
# FIG 09: PROBABILISTIC FORECAST INTERVALS
# ══════════════════════════════════════════════════════════════════════════
print('Generating Fig 09: Probabilistic Forecast Intervals...')

if predictions_available:
    df_sar = pd.read_parquet(f"{DATA_DIR}/06_sarimax_predictions.parquet")
    df_xgb = pd.read_parquet(f"{DATA_DIR}/07_xgboost_predictions.parquet")
    n_plot = min(48, len(df_sar))
    df_sar_plot = df_sar.tail(n_plot).reset_index(drop=True)
    df_xgb_plot = df_xgb.tail(n_plot).reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df_sar_plot['DATE'], df_sar_plot['y_true'], color='black', lw=2, label='Actual')
    axes[0].plot(df_sar_plot['DATE'], df_sar_plot['y_pred_sarimax'], color=COLORS['sarimax'], lw=1.8, label='SARIMAX+ONI')
    axes[0].fill_between(df_sar_plot['DATE'], df_sar_plot['pi_lower_95'], df_sar_plot['pi_upper_95'],
                         color=COLORS['sarimax'], alpha=0.25, label='95% PI')
    axes[0].set_ylabel('Y_stoch'); axes[0].set_title('(a) SARIMAX Probabilistic Forecast')
    axes[0].legend(loc='upper right')

    axes[1].plot(df_xgb_plot['DATE'], df_xgb_plot['y_true'], color='black', lw=2, label='Actual')
    axes[1].plot(df_xgb_plot['DATE'], df_xgb_plot['y_pred_xgb'], color=COLORS['xgb'], lw=1.8, label='XGBoost')
    axes[1].fill_between(df_xgb_plot['DATE'], df_xgb_plot['pi_lower_90'], df_xgb_plot['pi_upper_90'],
                         color=COLORS['xgb'], alpha=0.25, label='90% PI')
    axes[1].set_ylabel('Y_stoch'); axes[1].set_title('(b) XGBoost Probabilistic Forecast')
    axes[1].legend(loc='upper right')

    fig.suptitle('Probabilistic Forecasts with Prediction Intervals', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig09_probabilistic_forecast.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ✓ Fig 09 saved')
else:
    print('  ⚠ Fig 09 skipped: prediction files unavailable')

# ══════════════════════════════════════════════════════════════════════════════════
# FIG 10: SHAP FEATURE IMPORTANCE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print('Generating Fig 10: SHAP Summary...')

if shap_available:
    shap_summary = pd.read_csv(f"{OUT_DIR}/08_shap_feature_summary.csv")
    shap_summary = shap_summary.sort_values('mean_abs_SHAP', ascending=True)
    colors = [COLORS['shap_pos'] if m > 0 else COLORS['shap_neg'] for m in shap_summary['mean_SHAP'].values]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(shap_summary['Feature'], shap_summary['mean_abs_SHAP'], color=colors, alpha=0.85)
    ax.set_xlabel('Mean |SHAP|'); ax.set_title('SHAP Feature Importance Summary')
    for i, val in enumerate(shap_summary['mean_abs_SHAP']):
        ax.text(val + 0.0005, i, f"{val:.4f}", va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig10_shap_summary.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ✓ Fig 10 saved')
else:
    print('  ⚠ Fig 10 skipped: SHAP outputs unavailable')

# ══════════════════════════════════════════════════════════════════════════════════
# FIG 11: OLS-SHAP CORRESPONDENCE
# ══════════════════════════════════════════════════════════════════════════
print('Generating Fig 11: OLS-SHAP Correspondence...')

if shap_available:
    xai_corr = pd.read_csv(f"{OUT_DIR}/08_econometric_xai_correspondence.csv")
    xai_corr['sign_label'] = xai_corr['sign_concordance'].map({True: 'Agree', False: 'Flip'})
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(xai_corr['OLS_std_effect'], xai_corr['mean_abs_SHAP'],
                    c=xai_corr['sign_concordance'].map({True: COLORS['shap_pos'], False: COLORS['shap_neg']}),
                    s=80, alpha=0.85, edgecolor='k')
    for _, row in xai_corr.iterrows():
        ax.text(row['OLS_std_effect'], row['mean_abs_SHAP'], row['Feature'], fontsize=8,
                va='bottom', ha='right')
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.axvline(0, color='gray', lw=1, ls='--')
    ax.set_xlabel('OLS Standardized Effect'); ax.set_ylabel('Mean |SHAP|')
    ax.set_title('OLS vs SHAP Correspondence: Econometric Coefficients and Feature Impact')
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig11_ols_shap_correspondence.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ✓ Fig 11 saved')
else:
    print('  ⚠ Fig 11 skipped: SHAP outputs unavailable')

# ══════════════════════════════════════════════════════════════════════════════════
# FIG 13: ENSO PHASE FORECASTING
# ══════════════════════════════════════════════════════════════════════════
print('Generating Fig 13: ENSO Phase Forecasting...')

if predictions_available:
    df_meta = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")[['DATE','ENSO_phase']].copy()
    df_all = df_meta.merge(df_ols[['DATE','y_true','y_pred_ols']], on='DATE', how='inner')
    df_all = df_all.merge(df_sar[['DATE','y_pred_sarimax']], on='DATE', how='inner')
    df_all = df_all.merge(df_xgb[['DATE','y_pred_xgb']], on='DATE', how='inner')
    df_all = df_all[df_all['DATE'] >= pd.Timestamp('2015-01-01')]

    summary = []
    for phase, grp in df_all.groupby('ENSO_phase'):
        summary.append({
            'ENSO_phase': phase,
            'OLS_RMSE': np.sqrt(np.mean((grp['y_true'] - grp['y_pred_ols'])**2)),
            'SARIMAX_RMSE': np.sqrt(np.mean((grp['y_true'] - grp['y_pred_sarimax'])**2)),
            'XGB_RMSE': np.sqrt(np.mean((grp['y_true'] - grp['y_pred_xgb'])**2)),
            'n_months': len(grp)
        })
    df_phase = pd.DataFrame(summary).sort_values('ENSO_phase')

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_phase))
    width = 0.22
    ax.bar(x - width, df_phase['OLS_RMSE'], width, label='OLS-HC3', color=COLORS['ols'], alpha=0.85)
    ax.bar(x,         df_phase['SARIMAX_RMSE'], width, label='SARIMAX+ONI', color=COLORS['sarimax'], alpha=0.85)
    ax.bar(x + width, df_phase['XGB_RMSE'], width, label='XGBoost', color=COLORS['xgb'], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(df_phase['ENSO_phase'])
    ax.set_ylabel('RMSE'); ax.set_title('Forecast RMSE by ENSO Phase')
    ax.legend()
    for i, row in df_phase.iterrows():
        ax.text(i - width, row['OLS_RMSE'] + 0.002, f"{row['OLS_RMSE']:.3f}", ha='center', fontsize=8)
        ax.text(i, row['SARIMAX_RMSE'] + 0.002, f"{row['SARIMAX_RMSE']:.3f}", ha='center', fontsize=8)
        ax.text(i + width, row['XGB_RMSE'] + 0.002, f"{row['XGB_RMSE']:.3f}", ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig13_enso_phase_forecasting.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ✓ Fig 13 saved')
else:
    print('  ⚠ Fig 13 skipped: prediction files unavailable')

# ══════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════════
figs_generated = [f for f in os.listdir(FIG_DIR) if f.endswith('.png')]
print(f"\n✅ Notebook 10 complete.")
print(f"   Figures saved in: {FIG_DIR}")
print(f"   Total figures generated: {len(figs_generated)}")
for f in sorted(figs_generated):
    fpath = os.path.join(FIG_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"   {f:<50} ({size_kb:.0f} KB)")
