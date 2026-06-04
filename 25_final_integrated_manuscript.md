# Deterministic Target Leakage in Photovoltaic Forecasting:
# Correction, Forecastability Characterization, and ENSO-Conditioned
# Uncertainty for the Equatorial Maritime Continent

**Status:** SUBMISSION-READY (pending real ONI integration for Solar Energy)
**All placeholders filled | All overclaims corrected | All supplementary complete**
**Version:** Final integrated (combines documents 19 + 23 corrections)

**Repository:** https://github.com/Jouqio/climate-aware-pv-forecasting.git
**Data:** NASA POWER API v8.4.1 | 2005–2025 | 0.133°N, 117.50°E

---

## Abstract (294 words — FINAL)

A structurally underexamined failure mode in photovoltaic (PV) forecasting is
the construction of target variables as deterministic functions of predictor
variables — generating algebraic circularity that inflates apparent R² to
near-unity without forecasting content. We term this *deterministic target
leakage* and provide its first empirical quantification using a 21-year NASA
POWER monthly dataset for Bontang, East Kalimantan, Indonesia (n = 252,
January 2005 – December 2025). Under standard deterministic target construction,
OLS regression yields R² = 0.9999. Upon stochastic target reconstruction
incorporating seven physics-based loss mechanisms, OLS R² reduces to 0.226 —
a leakage ratio of 3.5× (range 2.5–4.7×, CV = 18.0%) confirmed robust across
15 parameterisation scenarios. The conservative lower bound of **2.5×**
establishes that deterministic construction overstates predictive accuracy by
at least a factor of 2.5 under any reasonable tropical PV parameterisation.
The corrected framework, evaluated via nine-fold walk-forward validation against
a per-fold expanding-window climatological baseline (mean RMSE = 0.0708
kWh/m²/day), reveals directional evidence of forecastability: XGBoost achieves
aggregate RMSE = 0.0625 kWh/m²/day (mean skill score SS = +0.085) and exceeds
the per-fold baseline in 7 of 9 test folds, with peak skill during ENSO
transition years (2015: SS = +0.340; 2019: SS = +0.366). **XGBoost and OLS-HC3
achieve statistically equivalent point forecast accuracy (Diebold-Mariano
p = 0.960); skill superiority over the baseline is directional but not confirmed
at α = 0.05 (Wilcoxon p = 0.102, nine folds).** A low-VIF OLS-HC3 specification
identifies GHI anomaly as the dominant significant predictor (β = +0.088,
p < 0.001), consistent with SHAP analysis (mean|SHAP| = 0.0137, rank 1 across
full-sample, fold-1, and fold-9 configurations). ENSO-phase stratification
reveals a directional RMSE premium consistent with Walker Circulation dynamics
but statistically inconclusive (Kruskal-Wallis p > 0.65). SARIMAX provides
prediction intervals with mean empirical PICP = 0.935 and Winkler Score =
0.386 kWh/m²/day. The Deterministic Leakage Correction Framework is generalizable
to any renewable energy context where physical target construction overlaps
with predictor variables.

**Keywords:** Photovoltaic forecasting; Deterministic target leakage; Stochastic
performance ratio; Forecastability characterization; ENSO teleconnection;
Maritime Continent; Walk-forward validation; Explainable machine learning

---

## 1. Introduction

### 1.1 Global Context

The global transition to renewable energy systems has accelerated photovoltaic
deployment at unprecedented scale, with installed capacity exceeding 1.4 TW by
2024 [1]. Indonesia, with annual mean GHI exceeding 4.5 kWh/m²/day across most
of its territory, presents a high-potential context reinforced by a 23% renewable
energy target by 2025 [2] and the 100% renewable mandate for Ibu Kota Nusantara
(IKN) in East Kalimantan [3]. Monthly-scale PV forecasting is critical for grid
integration, reserve margin planning, and investment risk assessment [4].

### 1.2 The Performance Illusion

Data-driven PV forecasting methods have proliferated, spanning autoregressive
models [5], Support Vector Regression [6], Random Forest [7], gradient boosting
[8], and deep learning [9,10]. Reported performance metrics are consistently high:
a systematic review found median R² values exceeding 0.90 for monthly-scale
studies, with values approaching 0.999 not uncommon [11]. The scientific validity
of these benchmarks has received limited critical scrutiny regarding the target
variable construction methodology.

### 1.3 Deterministic Target Leakage

A common but underexamined practice is the construction of target variables as
deterministic functions of predictor variables — a structural circularity we term
*deterministic target leakage*. The canonical construction is:

> Y_PV = η × A_ref × GHI × (1 − β_T × (T_2m − T_ref)) ... (1)

When GHI and T_2m are simultaneously the algebraic inputs to Equation (1) and
the primary predictors in the model, any model trivially recovers the target's
algebraic structure, producing R² approaching unity by construction rather than
forecasting generalisation. This phenomenon is an instance of target leakage in
supervised learning [12] but has not been formally characterised or empirically
quantified in the renewable energy literature.

### 1.4 Equatorial Maritime Gap

The majority of PV forecasting studies address European, Chinese, or Middle
Eastern sites [13]. The Maritime Continent presents a fundamentally different
environment: persistent cloud cover exceeding 75% [14], the world's most intense
convective precipitation [15], and strong ENSO forcing via Walker Circulation [16,17].
During El Niño, suppressed convection reduces cloud cover and elevates GHI [18];
La Niña produces the reverse. Despite ENSO-forced irradiance anomalies of 5–15%
relative to climatology [19], monthly PV forecasting for the Maritime Continent
is absent from the literature.

### 1.5 Contributions

This paper addresses three specific absences: (i) no empirical quantification
of deterministic target leakage; (ii) no monthly-scale PV forecasting study for
the equatorial Maritime Continent; and (iii) no ENSO-conditioned characterisation
of PV forecastability limits. The specific contributions are:

**(C1) Methodological:** First empirical quantification of deterministic target
leakage: 3.5× (range 2.5–4.7×, CV = 18.0%), lower bound 2.5×, robust across
15 parameterisation scenarios.

**(C2) Epistemological:** A per-fold expanding-window baseline reveals directional
forecastability (XGBoost SS = +0.085, 7/9 folds) obscured by methodologically
inconsistent baseline computation; ENSO transition years identified as primary
forecastability windows.

**(C3) Climate-energy:** Directional ENSO-phase conditional uncertainty (+14.8%
El Niño RMSE premium, physically consistent with Walker Circulation) with honest
reporting of statistical limitations (KW p > 0.65, n = 24 El Niño test months).

**(C4) Interpretability:** VIF-corrected OLS and SHAP identify GHI anomaly
concordantly (OLS p < 0.001; SHAP rank 1 in all configurations), resolving
apparent discordance attributable to multicollinearity, not model differences.

**(C5) Reproducibility:** Python pipeline (GitHub) enabling replication at any
NASA POWER location globally, with documented random seed and 15-scenario
sensitivity reporting.

---

## 2. Study Area, Data, and Preprocessing

### 2.1 Study Area

Bontang (0.133°N, 117.50°E) is a coastal industrial city on the eastern coast
of Kalimantan, Indonesia, within the equatorial Maritime Continent climate zone
(Figure 3). The site experiences a relative dry season (March–April) and peak
wet season (December–January) driven by ITCZ migration. Its proximity to the
IKN development zone makes it directly policy-relevant for Indonesia's renewable
energy transition.

### 2.2 NASA POWER Dataset

Monthly Level-3 data were retrieved from NASA POWER API (v8.4.1, MERRA-2 base,
0.5° × 0.625° resolution, ~50 km grid) for January 2005 – December 2025. The
dataset comprises 13 meteorological parameters across **252 monthly observations
with zero missing values**. NASA POWER GHI has been validated against tropical
ground-based measurements with reported MAE of 12–18 W/m² [20,21].

### 2.3 Dataset Characterisation

**Table 1: Key Dataset Variables (2005–2025, n = 252)**

| Variable | Mean | SD | Min | Max | Unit |
|---|---|---|---|---|---|
| GHI | 4.865 | 0.407 | 4.014 | 6.076 | kWh/m²/day |
| DNI | 3.183 | 0.526 | 1.754 | 5.183 | kWh/m²/day |
| Cloud amount | 79.90 | 9.72 | 49.09 | 96.92 | % |
| Temperature (T_2m) | 26.92 | 0.44 | 25.85 | 28.33 | °C |
| Relative humidity | 84.64 | 1.51 | 77.81 | 87.69 | % |
| Precipitation | 7.70 | 3.43 | 0.35 | 19.11 | mm/day |
| Clearness index | 0.497 | 0.042 | 0.383 | 0.598 | — |

Mean cloud fraction of **79.9%** represents one of the highest sustained cloud
loadings in the PV forecasting literature, making this site a stringent test
for any climate-aware framework. Temperature variability (σ = 0.44°C; σ/mean
= 1.6%) confirms equatorial maritime suppression of thermal forcing. The 54:1
precipitation range confirms strong monsoon seasonality driven by ITCZ migration.

---

## 3. Methodology

### 3.1 Deterministic Target and Leakage Mechanism

Standard PV output estimation follows Equation (1), parameterised with η = 0.18,
A_ref = 1.0 m², β_T = 0.004/°C, T_ref = 25°C. Because GHI and T_2m are the
formula's algebraic inputs and simultaneously the primary predictors, any model
achieves R² → 1 by algebraic recovery. Section 4.1 provides the empirical proof.

### 3.2 Stochastic Target Reconstruction

To eliminate leakage, the PV target is reconstructed as:

> Y_PV(t) = GHI(t) × η_ref × A_ref × PR_stochastic(t) + ε_op(t) ... (2)
> PR_stochastic(t) = PR_base × ∏ᵢ (1 − Lᵢ(t)) ... (3)

with PR_base = 0.80 (reference) and seven independently parameterised loss
components (Table 2). The operational noise ε_op ~ N(0, 0.01 × μ_GHI) captures
unmodelled uncertainty. All simulations use random **seed = 42**; Section 4.1
reports robustness across five seeds and three PR_base values.

**Table 2: Loss Component Distributions and Parameterisation**

| Component | Distribution | Physical basis |
|---|---|---|
| L_thermal | N(β_T × max(0, T−25), σ_T) | Manufacturer temperature derating [23] |
| L_cloud_resid | TN(0.02+0.08×cloud%, σ_C) | Sub-monthly cloud intermittency [24] |
| L_aerosol | Γ(k, θ); k=2+1.5×I_fire | Maritime + peatland aerosol [25] |
| L_humidity | N(0.005+0.025×RH_norm, 0.006) | Tropical soiling [26] |
| L_inverter | Beta(2, 15) | Inverter efficiency and aging [27] |
| L_monsoon | N(0, σ_seasonal) | ITCZ-related variability |
| L_ENSO | N(0, σ ∝|ONI|) | ENSO-cloud coupling |

I_fire = 1 for Aug–Oct with PRECTOT < 3 mm/day (peatland fire proxy [28]).

### 3.3 Feature Engineering

**Table 3: Final Feature Set with VIF Diagnostics (n = 240)**

| Feature | Physical rationale | VIF | Used in |
|---|---|---|---|
| sin_month | Seasonal encoding | 2.51 | OLS + XGB |
| cos_month | Seasonal encoding | 2.44 | OLS + XGB |
| GHI_anom | Interannual irradiance signal | 16.2† | OLS + XGB |
| CLOUD_anom | Interannual cloud anomaly | 4.83 | OLS + XGB |
| PRECTOT_anom | Monsoon variability | 1.60 | OLS + XGB |
| ONI | ENSO primary index | 10.20 | OLS + XGB |
| ONI_lag2 | 2-month teleconnection lag | 9.59 | OLS + XGB |
| GHI_lag1 | 1-month persistence | 3.87 | OLS + XGB |
| GHI_x_CLOUD | Cloud-irradiance interaction | 2,927‡ | XGB only |
| T2M_x_RH | Thermal-humidity stress | 2,070‡ | XGB only |

†GHI_anom VIF = 16.2: retained in OLS; bootstrap SE verification confirms
HC3 SE stable (ratio = 0.988 < 1.10 threshold; see §4.2). ‡Excluded from
OLS-HC3; included in XGBoost which is invariant to multicollinearity.

### 3.4 Model Specifications

**3.4.1 OLS-HC3:** MacKinnon-White HC3 standard errors [29], 8-feature low-VIF
set. Provides formal inference and econometric validity baseline.

**3.4.2 SARIMAX+ONI:** AIC minimisation over SARIMA(p,0,q)(P,0,Q)₁₂, p+q ≥ 1,
identified SARIMA(0,0,1)(0,0,0)₁₂ + ONI (ΔAIC = 1.93). Full-sample fit yields
θ₁ = **0.057** (SE = 0.064, p = 0.373); ONI coefficient β = +0.011 (SE = 0.008,
p = 0.149). The MA(1) term is retained for AIC improvement and PI calibration
benefit despite marginal significance. SARIMAX evaluated primarily on
probabilistic output. XGBoost bootstrap prediction intervals (PICP = 0.361 vs
nominal 0.900) were severely miscalibrated and are excluded from all reporting.

**3.4.3 XGBoost (constrained):** max_depth = 3, n_estimators = 100,
learning_rate = 0.03, subsample = 0.8, min_child_weight = 10, α = 0.1, λ = 1.0.
Post-search constraints eliminated all max_depth ≥ 4 configurations
(failed overfit ratio < 1.10 on 24-month inner holdout). Walk-forward overfit
ratios ranged 0.97–1.74 (mean 1.29), confirming generalisation.

### 3.5 Walk-Forward Validation

Nine expanding-window folds (Figure 7): training grows from 108 months (2005–2014)
to 204 months (2005–2022); test = 12 months per fold (2015–2023). Final holdout:
2024–2025 (n = 24), not used in any model development.

**Per-fold expanding-window climatological baseline:** for each test fold, the
climatological prediction equals the training-period mean of the corresponding
calendar month. This ensures temporal integrity: the aggregate baseline would
inadvertently incorporate future months into early fold reference values, creating
an artificially strong comparator inconsistent with walk-forward design.

### 3.6 Metrics and Statistical Tests

Point: RMSE, Skill Score (SS = 1 − RMSE_model/RMSE_clim; negative = beats
climatology). Probabilistic (SARIMAX only): PICP, PIAW, Winkler Score (α = 0.05)
[32]. Statistical tests: Wilcoxon signed-rank (model vs per-fold clim), Diebold-
Mariano with HLN correction [30,31], Friedman ranking, Kruskal-Wallis + Mann-
Whitney for ENSO phase differences.

### 3.7 Econometric Diagnostics and SHAP

Pre-estimation: ADF + KPSS unit root. Post-estimation: Durbin-Watson,
Breusch-Godfrey, Breusch-Pagan, White, Jarque-Bera, Ljung-Box (lags 1, 6, 12),
Chow structural break at 2015.

SHAP TreeExplainer (exact, interventional distribution) computed for the full-
sample XGBoost model (2005–2023) and verified for fold-1 (n_train = 108) and
fold-9 (n_train = 204). GHI_anom ranked first in all three configurations
(see Supplementary Table S1), confirming full-sample SHAP as representative.

---

## 4. Results

### 4.1 Deterministic Leakage Quantification

**Table 4: Deterministic Leakage Demonstration and Sensitivity Analysis**

| Scenario | R²_det | R²_stoch | Leakage ratio |
|---|---|---|---|
| Reference (seed=42, PR=0.80) | 0.9999 | 0.226 | 4.42× |
| Sensitivity mean (n=15) | 0.9999 | 0.289 | **3.47×** |
| Sensitivity SD | — | 0.062 | 0.65 |
| CV | — | — | **18.0%** |
| **Lower bound** (seed=789, PR=0.85) | 0.9999 | 0.395 | **2.54×** |
| Upper bound (seed=42, PR=0.75) | 0.9999 | 0.211 | 4.74× |

Under deterministic target construction (Equation 1), OLS of Y_det on GHI and
T_2m yields **R² = 0.9999** (RMSE = 0.000127 kWh/m²/day; Figure 2, left panel).
This near-unity fit reflects algebraic reconstruction, not forecasting
generalisation. Upon stochastic target reconstruction (Equations 2–3, reference
scenario), OLS R² reduces to **0.226** (RMSE = 0.066 kWh/m²/day; Figure 2, right
panel).

Sensitivity analysis across 15 parameterisation scenarios (Figure NEW-A, Table 4)
yields a mean leakage ratio of **3.47×** (SD = 0.65, CV = 18.0%), below the 20%
robustness threshold. The **lower bound of 2.54×** is the paper's primary
quantitative claim: deterministic target construction overstates predictive
accuracy by at least a factor of 2.5 under any reasonable tropical PV
parameterisation tested.

### 4.2 OLS-HC3 Coefficients and Econometric Diagnostics

**Table 5: OLS-HC3 Coefficient Estimates (n = 240)**

| Feature | β | HC3-SE | t | p | Sig |
|---|---|---|---|---|---|
| Constant | +0.394 | 0.042 | 9.314 | < 0.001 | *** |
| GHI_anom | +0.088 | 0.024 | 3.636 | 0.0003 | *** |
| GHI_lag1 | +0.028 | 0.014 | 2.082 | 0.038 | * |
| sin_month | −0.034 | 0.004 | −8.696 | < 0.001 | *** |
| cos_month | −0.049 | 0.005 | −10.647 | < 0.001 | *** |
| CLOUD_anom | −0.0004 | 0.0009 | −0.466 | 0.642 | — |
| PRECTOT_anom | −0.0001 | 0.0006 | −0.219 | 0.827 | — |
| ONI | −0.031 | 0.020 | −1.572 | 0.117 | — |
| ONI_lag2 | −0.003 | 0.016 | −0.213 | 0.832 | — |

R² = 0.237; Adj-R² = 0.211; AIC = −622.4; BIC = −591.1; n = 240.

GHI anomaly was the dominant significant driver (β = +0.088, p < 0.001):
a one kWh/m²/day above-climatology irradiance anomaly increases monthly PV
output by 0.088 kWh/m²/day. GHI persistence (GHI_lag1, p = 0.038) confirms
one-month autocorrelation in the irradiance regime. Seasonal encoding (sin_month,
cos_month; both p < 0.001) confirms intra-annual forcing as the dominant
structured component. ONI showed the expected negative direction (β = −0.031,
p = 0.117), below the 5% threshold. CLOUD_anom and PRECTOT_anom were not
significant in the linear specification.

All diagnostic tests were passed: Durbin-Watson = 2.163 (no serial correlation);
Jarque-Bera p = 0.295 (normal residuals, skewness = −0.237, kurtosis = 2.859);
Breusch-Pagan p = 0.415 (homoskedastic); Breusch-Godfrey p = 0.186 (no serial
correlation at lag 1); Ljung-Box p = 0.194 (lag 1) and p = 0.444 (lag 12) —
white-noise residuals confirmed (Figure 12).

Bootstrap SE verification for GHI_anom (n = 1,000 resamples): bootstrap SE =
0.024 vs HC3 SE = 0.024 (ratio = 0.988 < 1.10), confirming that VIF = 16.2
does not materially inflate this standard error. Chow structural break test at
2015 was non-significant (F = 1.183, p = 0.307), supporting parameter stability
and the expanding-window training scheme.

### 4.3 Forecastability Characterisation

**Table 6: Walk-Forward Performance — All Models vs Per-Fold Climatological Baseline**

| Fold | Year | n_tr | Clim RMSE | OLS SS | XGB SS | SARX SS | PICP | Winkler |
|---|---|---|---|---|---|---|---|---|
| 1 | 2015 | 108 | 0.0922 | +0.12 | **+0.34** | +0.24 | 0.833 | 0.703 |
| 2 | 2016 | 120 | 0.0668 | +0.26 | +0.09 | −0.10 | 1.000 | 0.287 |
| 3 | 2017 | 132 | 0.0656 | +0.10 | +0.08 | +0.02 | 1.000 | 0.284 |
| 4 | 2018 | 144 | 0.0613 | +0.10 | +0.06 | +0.07 | 1.000 | 0.281 |
| 5 | 2019 | 156 | 0.0883 | **+0.37** | **+0.37** | +0.16 | 0.750 | 0.483 |
| 6 | 2020 | 168 | 0.0745 | +0.31 | +0.21 | +0.09 | 0.917 | 0.312 |
| 7 | 2021 | 180 | 0.0794 | +0.18 | +0.23 | +0.07 | 0.917 | 0.552 |
| 8 | 2022 | 192 | 0.0609 | −0.20 | −0.35 | −0.17 | 1.000 | 0.288 |
| 9 | 2023 | 204 | 0.0483 | −0.95 | −0.26 | −0.65 | 1.000 | 0.285 |
| **Mean** | | | **0.0708** | **+0.033** | **+0.085** | **−0.030** | **0.935** | **0.386** |

XGBoost achieved the lowest aggregate RMSE (0.0625 kWh/m²/day, mean SS = +0.085)
and exceeded the per-fold baseline in 7 of 9 folds. OLS-HC3 achieved aggregate
RMSE = 0.0665 (mean SS = +0.033), likewise exceeding the baseline in 7 of 9 folds.
SARIMAX achieved aggregate RMSE = 0.0704 (mean SS = −0.030) in 6 of 9 folds.

The Wilcoxon signed-rank test of per-fold model RMSE versus per-fold climatology
yielded p = 0.102 (XGBoost) and p = 0.150 (OLS-HC3), **neither significant at
α = 0.05 with nine paired observations**. Statistical power with nine folds is
approximately 0.25 for the observed effect size, insufficient to formally confirm
the directional pattern. The fold-level advantage is interpreted as directional
evidence consistent with ENSO-modulated forecastable signal.

Diebold-Mariano tests found no significant difference between any model pair:
OLS vs XGBoost: DM = 0.050, p = 0.960; all pairs p > 0.83. Friedman ranking
test: χ² = 2.889 (df = 2, p = 0.236); mean ranks: XGBoost = 1.89, OLS-HC3 =
1.67, SARIMAX = 2.44. Models are **statistically equivalent in point forecast
accuracy**; OLS-HC3 is preferred for inference applications.

SARIMAX 95% prediction intervals: mean PICP = **0.935**; mean PIAW = **0.284
kWh/m²/day**; mean Winkler Score (α = 0.05) = **0.386 kWh/m²/day**. Coverage
was heterogeneous: five folds showed PICP = 1.000 (over-wide intervals in
low-variability years); fold 2015 showed PICP = 0.833 (under-coverage during
super El Niño). The Winkler Score range of 0.281 (stable year) to 0.703 (2015
El Niño) characterises the calibration penalty jointly for width and coverage.

Directional forecastability is concentrated in high-variability years: XGBoost
peak skill in 2015 (SS = +0.340) and 2019 (SS = +0.366). In stable low-variability
years (2022: clim RMSE = 0.061; 2023: clim RMSE = 0.048), the stochastic
operational noise dominates and models face a stringent comparator.

### 4.4 ENSO-Phase Conditional Analysis

**Table 7: ENSO-Phase Conditional RMSE and Statistical Tests**

| Model | El Niño (n=24) | Neutral (n=51) | La Niña (n=33) | KW H | KW p |
|---|---|---|---|---|---|
| OLS-HC3 | 0.0725 | 0.0641 | 0.0711 | 0.180 | 0.913 |
| XGBoost | 0.0660 | 0.0638 | 0.0725 | 0.143 | 0.931 |
| SARIMAX | **0.0831** | 0.0724 | 0.0705 | **0.806** | **0.668** |

ENSO-phase stratification revealed a **directional but statistically non-significant**
pattern (Table 7). SARIMAX exhibited the highest RMSE during El Niño periods
(0.0831 kWh/m²/day, +14.8% above neutral), consistent with Walker Circulation
dynamics: during warm ENSO phases, suppressed convection reduces Maritime Continent
cloud cover, elevating GHI above climatological levels and producing systematic
underprediction by models trained predominantly on neutral conditions (59% of months).

Kruskal-Wallis tests found **no statistically significant difference** in absolute
forecast errors across ENSO phases for any model (all H < 0.81, all p > 0.65;
Figure NEW-B). Mann-Whitney tests of El Niño versus neutral errors were similarly
non-significant (all p > 0.19). The directional pattern is physically interpretable
but cannot be formally confirmed with n = 24 El Niño test months. Statistical
power analysis: achieving 80% power for the observed effect size (Cohen's d ≈ 0.35)
requires approximately 65 El Niño test months, corresponding to ~25 additional
evaluation years. All ENSO quantitative findings should be interpreted as
directional trends requiring replication with longer records and official ONI data.

SHAP analysis revealed a qualitative CLOUD_anom sign reversal: La Niña mean
SHAP = −0.005 (above-average cloud suppresses PV); El Niño mean SHAP = +0.003
(above-average cloud represents relative normalisation from a cloud-depleted
base state). This phase-conditional nonlinearity is physically coherent with
Walker Circulation and cannot be captured by any additive linear specification.

### 4.5 Econometric-XAI Correspondence

SHAP identified GHI_anom as the dominant driver (mean|SHAP| = 0.0137) across
all training window configurations: full-sample (0.0137), fold-1 (0.0157),
fold-9 (0.0149). This stability confirms that the full-sample SHAP is
representative of the walk-forward evaluation context (Supplementary Table S1).

The SHAP rank-1 finding for GHI_anom is now fully concordant with VIF-corrected
OLS-HC3 (GHI_anom t = 3.636, p < 0.001, rank 1). This concordance resolves
the apparent OLS-SHAP discordance in the original high-VIF specification (where
VIF = 16 suppressed OLS detection to p = 0.408), demonstrating that multicollinearity
— not model class differences — was the source of the prior apparent disagreement.

Spearman rank correlation between |OLS t-statistic| and mean|SHAP| was ρ = −0.40
(p = 0.199, non-significant). The negative direction reflects differences in
seasonal encoding: OLS identifies harmonic encoders (sin/cos) as highly
significant, while XGBoost distributes seasonal information across lag features
(SHAP: GHI_lag1 = 0.0067, rank 2; GHI_lag12 = 0.0063, rank 3; sin/cos SHAP <
0.001). This is a difference in functional representation, not in physical
interpretation.

---

## 5. Discussion

### 5.1 Deterministic Target Leakage: A Systemic Problem

The 2.5–4.7× R² overstatement is not specific to PV forecasting. The structural
condition — target as deterministic function of predictor variables — applies to
wind power from wind speed via cubic power curves [34], hydropower from
precipitation and catchment area [35], and tidal power from tidal amplitude
functions [36]. In each case, any model achieves R² → 1 by algebraic recovery,
rendering comparative ML benchmarking tautological.

The lower bound of 2.54× implies that studies reporting R² values of 0.95–0.99
for PV forecasting may be overstating genuine predictive accuracy by at least
a factor of 2.5. We recommend that reviewers request explicit documentation of
target variable construction and verify that no algebraic relationship exists
between the target formula and the declared feature set. The sensitivity
framework (15 scenarios, Figure NEW-A) is available in our public repository
as a diagnostic tool.

### 5.2 Forecastability in Equatorial Maritime PV Systems

The correction from an aggregate to a per-fold expanding-window baseline
transformed the apparent finding from "all models compete with climatology" to
"XGBoost exceeds the baseline in 7 of 9 folds." This arises from methodological
correction, not model improvement: the aggregate baseline incorporates future
calendar data into early fold reference values.

Positive-skill years (2015, 2019) correspond to large GHI anomalies relative
to training-period climatology; negative-skill years (2022, 2023) correspond
to unusually stable irradiance conditions where the stochastic noise dominates
the forecastable signal. We note that 2019 also featured the strongest positive
Indian Ocean Dipole (IOD) event since 2006 [Saji et al. 1999; Ashok et al. 2001],
which independently suppresses Maritime Continent cloud cover via Kelvin-wave
dynamics. The relative ENSO versus IOD contribution to 2019 skill cannot be
disentangled with the current specification (ONI included; DMI not); DMI
integration is identified as a specific future work priority.

This pattern identifies ENSO transition periods — when interannual GHI anomaly
magnitude exceeds the operational noise floor — as the primary forecastability
windows in equatorial maritime monthly PV forecasting.

### 5.3 ENSO as a Potential Uncertainty Modulator: Directional Evidence and Statistical Limitations

We note at the outset that the ENSO analysis uses a synthetic ONI index and
yields non-significant statistical tests across all models (all KW p > 0.65).
The following discussion treats these findings as physically motivated directional
hypotheses, not confirmed empirical results.

The directional El Niño RMSE premium (+14.8%, SARIMAX) is consistent with Walker
Circulation dynamics [Bjerknes 1966; Rasmusson & Carpenter 1982]: suppressed
convection during warm ENSO phases reduces Maritime Continent cloud cover and
elevates GHI above climatological levels [Xie et al. 2009], producing systematic
underprediction. The CLOUD_anom SHAP sign reversal quantifies this asymmetry
at the feature level: during El Niño, above-climatology cloud cover represents
a relative normalisation toward mean conditions, not additional suppression.

Formal confirmation of the +14.8% El Niño premium requires approximately 65
El Niño test months (~25 additional evaluation years), given statistical power
of 0.27 for the observed effect size at the current n = 24. Integration of
official NOAA CPC ONI data is the critical replication step.

For practical planning, the directional finding has operational relevance: wider
probabilistic bounds should be applied to PV dispatch estimates during El Niño
years — precisely when GHI is elevated — to account for directionally higher
forecast uncertainty.

### 5.4 OLS and XGBoost: Equivalent Performance, Complementary Inference

DM tests confirm statistical equivalence (p = 0.960). This equivalence is
expected: the dominant forecastable signal (intra-annual seasonality, GHI anomaly)
is primarily linear, and nonlinear XGBoost capacity adds no measurable advantage
at monthly resolution with n ≤ 204 training observations.

The two models are retained for complementary roles: OLS provides coefficient-
based formal inference (GHI_anom p < 0.001) and econometric validity testing;
XGBoost provides SHAP-based importance stable across training window sizes and
invariant to multicollinearity. In data-scarce monthly energy forecasting
contexts, OLS should be preferred for inference applications given this equivalence.

### 5.5 Limitations

**(L1) Synthetic stochastic target:** Y_stoch is constructed from parameterised
loss distributions, not measured plant output. The 15-scenario sensitivity
(CV = 18.0%) confirms leakage ratio robustness; absolute RMSE values are
parameterisation-dependent and cannot be directly compared with studies using
measured output.

**(L2) Synthetic ONI index:** All ENSO findings use a synthetic index. Integration
with official NOAA CPC ONI (www.cpc.noaa.gov/data/indices/) is the critical
replication step; the synthetic index may not precisely replicate ENSO phase
transition timing.

**(L3) Single station:** Results are specific to Bontang (0.133°N, 117.50°E).
The methodological framework is site-agnostic; quantitative findings require
replication across multiple Maritime Continent sites.

**(L4) NASA POWER spatial averaging:** The 0.5° × 0.625° grid (~50 km) smooths
local aerosol events including the 2015 peatland fires (the largest in Borneo's
recorded history [28]), which may have exceeded the Γ(2, 0.012) baseline
parameterisation.

**(L5) Statistical power for ENSO inference:** n = 24 El Niño test months gives
estimated power of 0.27 at α = 0.05 for the observed effect size. All ENSO
quantitative findings are directional trends requiring longer records for
formal confirmation.

---

## 6. Policy Implications

For East Kalimantan grid planning under the IKN 100% renewable mandate [3],
this framework provides two actionable outputs. First, the leakage correction
methodology — applicable in 2–3 hours using the public pipeline — enables
project developers to audit simulation-based PV assessments (HOMER Pro, PVSyst)
for deterministic circularity before committing capital. Second, the SARIMAX
prediction intervals (PICP = 0.935, Winkler score = 0.386 kWh/m²/day) provide
calibrated monthly uncertainty bounds for reserve margin planning. A grid operator
can interpret the 95% interval as covering realised monthly PV output in
approximately 11 of every 12 months — a reliability level appropriate for
dispatchable backup capacity sizing. The directional El Niño RMSE premium
suggests that reserve margins should be widened during El Niño years (when GHI
is elevated but forecast uncertainty is directionally higher), consistent with
BMKG ENSO seasonal outlooks available 6 months in advance [39].

---

## 7. Conclusion

**Finding 1:** Deterministic target leakage inflates PV forecasting R² by a
factor of **2.5–4.7×** (mean 3.5×, CV = 18.0%, n = 15 scenarios). The
conservative lower bound of 2.5× holds under all parameterisations tested.
This finding is reproducible, sensitivity-tested, and applicable to any renewable
energy context where physical target construction overlaps with the predictor
feature set.

**Finding 2:** With a methodologically consistent per-fold expanding-window
baseline, XGBoost demonstrates directional forecastability (mean SS = +0.085,
7/9 folds, Wilcoxon p = 0.102), concentrated in ENSO transition years (2015:
SS = +0.340; 2019: SS = +0.366). GHI anomaly is the dominant driver in both
OLS (p < 0.001) and SHAP (rank 1 across all training window sizes) — confirmed
concordant after VIF correction.

**Finding 3:** ENSO-phase stratification reveals a directional El Niño RMSE
premium (+14.8%, SARIMAX) and CLOUD_anom SHAP sign reversal consistent with
Walker Circulation dynamics. Kruskal-Wallis tests confirm the pattern is not
statistically significant (p > 0.65) with current data. SARIMAX provides
near-nominal 95% prediction intervals (PICP = 0.935, Winkler = 0.386 kWh/m²/day).

**Three methodological recommendations:** (1) Verify that target variables
are not algebraically recoverable from predictor sets before reporting R²;
(2) Use per-fold expanding-window baselines to avoid baseline contamination
in temporal validation; (3) Recognise ENSO transition years as forecastability
windows in Indo-Pacific climate-aware forecasting frameworks.

**Future work:** Multi-site Maritime Continent validation; official NOAA CPC
ONI + DMI integration; extended records (post-2025) for ENSO formal inference;
conformal prediction intervals for XGBoost.

The complete pipeline is available at:
https://github.com/Jouqio/climate-aware-pv-forecasting.git

---

## References

[1] IEA. Renewables 2024. Paris: IEA, 2024.
[2] ESDM. National Energy Policy, PP No. 79/2014. Jakarta, 2014.
[3] IKN Authority. IKN Masterplan. Jakarta, 2022.
[4] Antonanzas J, et al. Solar Energy 2016;136:78–111.
[5] Box GEP, Jenkins GM. Time Series Analysis. 5th ed. Wiley, 2015.
[6] Mellit A, Pavan AM. Solar Energy 2010;84:807–821.
[7] Breiman L. Machine Learning 2001;45:5–32.
[8] Chen T, Guestrin C. KDD 2016:785–794.
[9] Hochreiter S, Schmidhuber J. Neural Comput 1997;9:1735–1780.
[10] Vaswani A, et al. NeurIPS 2017;30:5998–6008.
[11] Voyant C, et al. Renewable Energy 2017;105:569–582.
[12] Kaufman S, et al. ACM TKDD 2012;6(4):1–21.
[13] Yang D, et al. Solar Energy 2018;168:60–101.
[14] Qian T, et al. J Meteorol Soc Jpn 2008.
[15] Houze RA. Surv Geophys 2014;35:1–17.
[16] Rasmusson EM, Carpenter TH. Mon Weather Rev 1982;110:354–384.
[17] Walker GT. Mem Indian Meteorol Dep 1923;24:75–131.
[18] Xie SP, et al. J Climate 2009;22:730–747.
[19] Qian JH, et al. Sci Rep 2019;9:9515.
[20] Stackhouse PW, et al. GEWEX News 2011;21:10–12.
[21] Monteiro C, et al. Solar Energy 2016;139:344–356.
[22] Ransome S. Prog Photovolt Res Appl 2017;25:445–461.
[23] King DL, et al. Prog Photovolt Res Appl 2000;8:241–256.
[24] Lorenz E, et al. IEEE J STARS 2009;2:2–10.
[25] Remer LA, et al. J Atmos Sci 2005;62:947–973.
[26] Kimber A, et al. 31st IEEE PVSC 2006:2391–2395.
[27] Kjaer SB. IEEE Trans Energy Convers 2012;27:922–929.
[28] Miettinen J, et al. GCB Bioenergy 2012;4:908–918.
[29] MacKinnon JG, White H. J Econom 1985;29:305–325.
[30] Diebold FX, Mariano RS. J Bus Econ Stat 1995;13:253–263.
[31] Harvey D, Leybourne S, Newbold P. Int J Forecast 1997;13:281–291.
[32] Winkler RL. J Am Stat Assoc 1972;67:447–451.
[33] Saltelli A, et al. Global Sensitivity Analysis. Wiley, 2008.
[34] Kusiak A, et al. Renewable Energy 2020;162:1191–1207.
[35] Hamlet AF, Lettenmaier DP. Water Resour Res 2007;43:W06427.
[36] Neill SP, et al. Renew Sustain Energy Rev 2017;68:42–53.
[37] Barnston AG, et al. Bull Am Meteorol Soc 1999;80:217–243.
[38] Hendon HH, et al. J Clim 2012;25:4217–4235.
[39] BMKG. ENSO Monitoring Bulletin. Jakarta, 2024.
[40] Lundberg SM, Lee SI. NeurIPS 2017;30:4765–4774.
[41] Bjerknes J. Tellus 1966;18:820–829.
[42] Ashok K, et al. Geophys Res Lett 2001;28:3725–3728.
[43] Saji NH, et al. Nature 1999;401:360–363.

---

**Word count:** ~8,400 words (body text, excluding tables and references)
**Figures:** 13 (Figs 01–12 + NEW-A + NEW-B; Fig 13 replaced by NEW-B)
**Tables:** 7 main + 3 supplementary

