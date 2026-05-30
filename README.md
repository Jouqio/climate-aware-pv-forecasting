# Stochastic Climate-Aware Probabilistic PV Forecasting Framework

## Bontang, East Kalimantan, Indonesia | NASA POWER 2005–2025

[![Target Journal](https://img.shields.io/badge/Target-Applied%20Energy%20Q1-blue)]()
[![Data](https://img.shields.io/badge/Data-NASA%20POWER%20v8.2.1-green)]()
[![Period](<https://img.shields.io/badge/Period-2005--2025%20(252%20months)-orange>)]()
[![Location](https://img.shields.io/badge/Location-0.1333°N%20117.50°E-red)]()

---

## Overview

This repository implements a **Scopus Q1/Q2 research-grade pipeline** for
photovoltaic power forecasting at Bontang, East Kalimantan (near IKN Nusantara),
Indonesia. It addresses two fundamental limitations of existing tropical PV studies:

1. **Deterministic Leakage Problem** — algebraic target variables inflating R² to
   near-unity (R² > 0.999) with no genuine predictive content.
2. **Climate Teleconnection Gap** — failure to model ENSO/IOD/MJO influence on
   equatorial maritime solar resource variability.

### Key Contributions

| #   | Contribution                                     | Type             |
| --- | ------------------------------------------------ | ---------------- |
| 1   | Deterministic Leakage Correction Protocol (DLCP) | Methodological   |
| 2   | 9-Component Stochastic PV Target with Borneo AOD | Methodological   |
| 3   | 21-year multi-decadal NASA POWER panel (n=252)   | Data             |
| 4   | ENSO-Stratified 5-Fold Walk-Forward Validation   | Validation       |
| 5   | OLS-SHAP Econometric-XAI Correspondence Bridge   | Interpretability |
| 6   | IKN Nusantara P10/P50/P90 Energy Planning        | Policy           |

---

## Pipeline Architecture

```
00_main_pipeline.py          ← Run all modules in sequence
│
├── 01_data_loader.py        ← Load NASA POWER CSV + QA + ENSO labels
├── 02_stochastic_target.py  ← 9-component stochastic PV reconstruction
├── 03_feature_engineering.py← 35-feature pipeline + LASSO selection
│
├── 04_econometric_models.py ← TIER I:  OLS-HC3, GLS, Ridge, LASSO, QR, Bayesian
├── 05_timeseries_models.py  ← TIER II: SARIMA, SARIMAX, Prophet, VAR, DHR
├── 06_ml_models.py          ← TIER III: RF, XGBoost, LightGBM, CatBoost, SVR
├── 07_deep_learning.py      ← TIER IV: LSTM, GRU, TFT-Lite, N-BEATS
├── 08_probabilistic.py      ← TIER V:  QRF, GPR, Conformal, Ensemble
│
├── 09_diagnostics.py        ← 18-test statistical diagnostic protocol
├── 10_climate_teleconnection.py ← ENSO/IOD/MJO analysis + Granger causality
├── 11_explainable_ai.py     ← SHAP global+local + OLS-SHAP correspondence
├── 12_validation.py         ← ENSO-stratified walk-forward + DM test
├── 13_visualization.py      ← All mandatory figures (Fig 1–12)
└── 14_results_compiler.py   ← Master tables + reviewer checklist
```

---

## Dataset

**File:** `POWER_Point_Monthly_20050101_20251231_000d13N_117d50E_UTC.csv`

| Property           | Value                                         |
| ------------------ | --------------------------------------------- |
| Source             | NASA POWER v8.2.1                             |
| Location           | 0.1333°N, 117.50°E (Bontang, East Kalimantan) |
| Period             | January 2005 – December 2025                  |
| n                  | 252 months                                    |
| Primary variable   | ALLSKY_SFC_SW_DWN (GHI, kWh/m²/day)           |
| GHI range          | 4.014 – 6.076 kWh/m²/day                      |
| GHI mean ± std     | 4.865 ± 0.407 kWh/m²/day (CV=8.36%)           |
| CLOUD mean ± std   | 79.9 ± 9.7% (CV=12.17%)                       |
| PRECTOT mean ± std | 7.70 ± 3.43 mm/day (CV=44.55%)                |

---

## Stochastic Target Formula

```
Y_stoch(t) = GHI(t) × η_STC
             × [1 − L_soil(t)]          # monsoon-modulated soiling
             × [1 − L_spectral(t)]      # cloud+RH spectral mismatch
             × η_inv(p, t)              # non-linear inverter + clipping
             × R_deg(t)                 # progressive degradation (21yr)
             × [1 − L_wire]             # DC+AC wiring losses
             × ξ_intermit(t)            # sub-daily MJO intermittency
             × [1 − β_eff(T_cell−25)]  # NOCT temperature coefficient
             × [1 − L_AOD(t)]           # ★ Borneo biomass burning AOD
             × δ_bias(t)               # ★ NASA POWER satellite bias
```

Target validation: **E[PR] ∈ [0.72, 0.80]** (IEA PVPS Task 13 tropical bounds)

---

## ENSO-Stratified Walk-Forward Validation

| Fold   | Training Period   | Validation Period | ENSO Context               |
| ------ | ----------------- | ----------------- | -------------------------- |
| Fold 1 | 2005–2012 (n=96)  | 2013–2014 (n=24)  | El Niño onset 2014         |
| Fold 2 | 2005–2014 (n=120) | 2015–2016 (n=24)  | **Strong El Niño 2015–16** |
| Fold 3 | 2005–2016 (n=144) | 2017–2019 (n=36)  | Neutral + extreme IOD 2019 |
| Fold 4 | 2005–2019 (n=180) | 2020–2022 (n=36)  | **Triple La Niña 2020–23** |
| Fold 5 | 2005–2022 (n=216) | 2023–2025 (n=36)  | El Niño 2023–24            |

---

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/pv-forecasting-bontang.git
cd pv-forecasting-bontang

# Install dependencies
pip install -r requirements.txt

# Optional: Deep Learning
pip install tensorflow

# Optional: Explainable AI
pip install shap

# Place NASA POWER CSV in project root, then run:
python 00_main_pipeline.py
```

---

## Outputs

```
outputs/
├── data/                    # All intermediate and final CSV results
│   ├── 01_processed_dataset.csv
│   ├── 02_stochastic_pv_target.csv
│   ├── 02_monte_carlo_results.csv
│   ├── 03_selected_features.csv
│   ├── 04_econometric_metrics.csv
│   ├── 05_timeseries_metrics.csv
│   ├── 06_ml_metrics.csv
│   ├── 07_dl_metrics.csv
│   ├── 08_probabilistic_metrics.csv
│   ├── 08_P10_P50_P90_exceedance.csv
│   ├── 09_diagnostic_results.csv
│   ├── 10_enso_conditional_stats.csv
│   ├── 11_ols_shap_correspondence.csv
│   ├── 12_walkforward_fold_metrics.csv
│   ├── 12_diebold_mariano_stats.csv
│   └── 14_master_model_comparison.csv
│
├── figures/                 # All publication figures (PNG, 180-200 dpi)
│   ├── 01_study_area_map.png
│   ├── 02_methodology_flowchart.png
│   ├── 03_timeseries_panel.png
│   ├── 04_stochastic_target_validation.png
│   ├── 05b_ols_diagnostic_4plot.png
│   ├── 06_climate_teleconnection.png
│   ├── 07_ml_model_comparison.png
│   ├── 08_walkforward_validation.png
│   ├── 09a_shap_beeswarm.png
│   ├── 09b_shap_importance_bar.png
│   ├── 10_probabilistic_evaluation.png
│   ├── 11_ikn_policy_p10p50p90.png
│   └── 12_loss_sensitivity_analysis.png
│
└── tables/                  # Publication-ready CSV tables
    ├── Table3_monte_carlo_pr_validation.csv
    ├── Table4_feature_importance.csv
    ├── Table5_diagnostics.csv
    ├── Table6_model_comparison.csv
    ├── Table7_enso_performance.csv
    └── Table8_probabilistic.csv
```

---

## Target Journals

| Journal                                | Quartile | IF (2024) | Priority     |
| -------------------------------------- | -------- | --------- | ------------ |
| Applied Energy                         | Q1       | ~11.0     | **Primary**  |
| Renewable & Sustainable Energy Reviews | Q1       | ~16.0     | Alternative  |
| Solar Energy                           | Q2       | ~5.0      | **Fallback** |
| Energy Conversion & Management         | Q1       | ~9.9      | Alternative  |

---

## Citation

```bibtex
@article{[author]2025stochastic,
  title   = {Stochastic Climate-Aware Probabilistic Photovoltaic Forecasting
             for Equatorial Maritime Indonesia: A 21-Year Multi-Decadal Framework
             Integrating Physics-Informed Loss Reconstruction, ENSO Teleconnection
             Modeling, and Explainable Machine Learning},
  author  = {[Author(s)]},
  journal = {Applied Energy},
  year    = {2025},
  note    = {Under review}
}
```

---

## Data Availability

- NASA POWER: https://power.larc.nasa.gov/
- ONI index: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
- DMI index: https://www.jamstec.go.jp/aplinfo/sintexf/iod/DMI.monthly.txt
- MJO index: http://www.bom.gov.au/climate/mjo/
- Dataset archived on: [Zenodo DOI — add after upload]

---

_Random seed: 42 — All stochastic experiments reproducible via np.random.seed(42)_
