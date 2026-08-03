<div align="center">

# ☀️ Climate-Aware Multi-Site PV Forecasting
### 10 Indonesian Locations

**Deterministic target leakage quantification and climate-aware
forecastability assessment for photovoltaic (PV) output across
ten locations spanning Indonesia's climatically diverse archipelago.**

[![Sites](https://img.shields.io/badge/Sites-10-2ea44f?style=for-the-badge)]()
[![Climate Regimes](https://img.shields.io/badge/Climate%20Regimes-6-blue?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-FC3D21?style=flat-square&logo=nasa&logoColor=white)](https://power.larc.nasa.gov)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=flat-square&logo=xgboost&logoColor=white)]()
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=flat-square)]()
[![Reproducible](https://img.shields.io/badge/Seed-42-success?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

</div>

---

##  Study Locations

**10 sites · 6 climate regimes** across the Indonesian archipelago

| City | Province | Regime | Leakage Ratio | GHI_anom β (p) |
|:---|:---|:---|:---:|:---:|
|  Medan | N. Sumatra | Equatorial Wet | 2.71× | +0.125 (<0.0001) |
|  Pekanbaru | Riau | Equatorial Wet | 3.05× | +0.112 (<0.0001) |
|  Pontianak | W. Kalimantan | Equatorial Maritime | 🔴 **3.43× (max)** | +0.104 (<0.0001) |
|  Bontang | E. Kalimantan | Equatorial Maritime | 2.98× | +0.087 (0.0017) |
|  Samarinda | E. Kalimantan | Equatorial Maritime | 2.56× | +0.091 (0.0002) |
|  Balikpapan | E. Kalimantan | Equatorial Maritime | 3.11× | +0.107 (<0.0001) |
|  Makassar | S. Sulawesi | Monsoonal | 🟢 **1.72× (min)** | +0.084 (0.027) |
|  Surabaya | E. Java | Tropical Monsoon | 2.21× | +0.088 (0.0037) |
|  Kupang | E. Nusa Tenggara | Semi-Arid Tropical | 1.87× | +0.075 (0.019) |
|  Jayapura | Papua | Equatorial Rainforest | 3.19× | +0.079 (0.0004) |

---

##  Key Cross-Site Findings

> All findings verified from real pipeline output — no simulated results.

| # | Finding |
|:---:|:---|
| 1️⃣ | **Deterministic leakage is universal** — R²_det ≈ 1.0000 at all 10 sites |
| 2️⃣ | **Leakage ratio range: 1.72×–3.43×**, systematically tied to local GHI variability (Spearman ρ = −0.903, permutation p = 0.0007, n = 10) |
| 3️⃣ | **Meta-analytic GHI_anom β = +0.102** (z = 15.28, p ≪ 10⁻⁶, **I² = 0.0%**) — quantitatively identical effect across 6 climate regimes |
| 4️⃣ | **XGBoost beats climatology at 3/10 sites** (Wilcoxon p < 0.05); between-site variation **not** statistically confirmed (KW p = 0.437) |
| 5️⃣ | **SHAP rank-1 = GHI_anom** at equatorial sites; **T2M × RH** dominates at lower-cloud, higher-variability sites (Makassar, Surabaya, Kupang) |

---

##  Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

#  Auto-discovers all data/raw/*.csv, runs the full pipeline
python run_all_locations.py

#  Run specific sites only
python run_all_locations.py --sites bontang makassar kupang

#  Partial pipeline (steps 1–5)
python run_all_locations.py --steps 1-5

#  Regenerate cross-site comparison only
python scripts/cross_site_comparison.py
```

> 💡 **Adding a new site:** drop `newcity.csv` into `data/raw/` and re-run —
> **zero code changes required** (auto-discovery).

---

##  Repository Structure

```
📦 climate-aware-pv-forecasting
│
├── 📂 data/
│   └── raw/                           10 NASA POWER CSVs (tracked)
│       ├── medan.csv
│       ├── pekanbaru.csv
│       ├── pontianak.csv
│       ├── bontang.csv
│       ├── samarinda.csv
│       ├── balikpapan.csv
│       ├── makassar.csv
│       ├── surabaya.csv
│       ├── kupang.csv
│       └── jayapura.csv
│
├── 📂 notebooks/                      10 location-agnostic pipeline scripts (01–10)
│
├── 📂 scripts/
│   ├── cross_site_comparison.py       auto-aggregates all discovered sites
│   ├── fetch_oni_dmi.py               optional: download real ONI/DMI ENSO indices
│   └── oni_integration_complete.py    optional: ONI integration utility
│
├── 📂 results/
│   ├── medan/        {data,outputs,figures}/
│   ├── pekanbaru/     {data,outputs,figures}/
│   ├── pontianak/    {data,outputs,figures}/
│   ├── bontang/      {data,outputs,figures}/
│   ├── samarinda/    {data,outputs,figures}/
│   ├── balikpapan/   {data,outputs,figures}/
│   ├── makassar/     {data,outputs,figures}/
│   ├── surabaya/     {data,outputs,figures}/
│   ├── kupang/       {data,outputs,figures}/
│   ├── jayapura/     {data,outputs,figures}/
│   └── combined/     cross-site metrics + 7 tables + 1 Excel + 12 figures
│
├── 📂 docs/
│   ├── DATA_STRUCTURE.md              data/raw/ convention, NASA POWER export settings
│   ├── EXPERIMENT_WORKFLOW.md         what each notebook does, in what order
│   ├── MULTI_SITE_EXECUTION.md        PV_LOCATION mechanics, auto-discovery, adding sites
│   ├── REPRODUCIBILITY.md             seeds, environment, exact replication steps
│   ├── MIGRATION_GUIDE.md             5-phase architecture history
│   ├── manuscript_10sites/            full manuscript (Abstract → Conclusion)
│   └── archive/single_site_era/       historical single-site working documents
│
├── 📄 utils.py                        shared leakage-free metrics (single source of truth)
├── 📄 run_all_locations.py            main entry point (auto-discovery)
├── 📄 run_pipeline.py                 legacy single-site runner (backward compat)
├── 📄 requirements.txt                pipeline dependencies
├── 📄 .gitignore                      excludes cache, env, and generated artifacts
├── 📄 LICENSE                         MIT License
└── 📄 README.md                       you are here
```

> 📌 **Project history & audit trail** (root-level docs from the revision process):
> `24_cover_letter_and_response.md` · `27_oni_integration_complete.py` ·
> `29_NEXT_STEPS_ACTION_CARD.md` · `30_major_revision_complete.md` ·
> `31_revision_changelog.md` · `32_final_project_summary.md` ·
> `39_CODE_AUDIT_CRITICAL_FINDINGS.md` · `40_notebook_patches_GUIDE.py` ·
> `42_q1_leakage_fix.patch` · `43_HOW_TO_APPLY_AND_PUSH.md`

---

##  Reproducibility

| Aspect | Detail |
|:---|:---|
|  Random seed | `np.random.seed(42)` for all stochastic components |
|  Validation | 9-fold walk-forward evaluation (test 2015–2023); 2024–2025 holdout unused |
|  Leakage check | `utils.verify_no_feature_leakage()` runs every fold at every site |
|  Verified runs | 90/90 notebook executions verified successful (10 sites × 9 steps) |

---

##  Data Source

<table>
<tr>
<td><b>Source</b></td>
<td><a href="https://power.larc.nasa.gov">NASA POWER Monthly API</a></td>
</tr>
<tr>
<td><b>Underlying data</b></td>
<td>MERRA-2 / CERES, 0.5° × 0.625° resolution</td>
</tr>
<tr>
<td><b>Community / Settings</b></td>
<td>Renewable Energy community, UTC, 2005–2025</td>
</tr>
</table>

---
## License

MIT License see LICENSE file.

<div align="center">

## Contact

Syauqi Nuzul Abdi | nuzulabdisyauqi@gmail.com 

</div>
