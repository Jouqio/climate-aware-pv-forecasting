#  NEXT STEPS — ACTION CARD
## What to do right now, in order

---

## STEP 1 — ONI Integration (2–3 hours) — Do this TODAY

```bash
# 1. Run the ONI integration script
python3 27_oni_integration_complete.py

# 2. Re-run pipeline from NB03 onwards
python3 run_pipeline.py --from 3

# 3. Record the new numbers:
cat outputs/09b_test_statistics.csv      # → new KW H and p
cat outputs/06_enso_phase_analysis.csv  # → new ENSO RMSE values
cat outputs/06b_sarimax_coefficients.csv # → new ONI coefficient
```

**In the manuscript (doc 25), find all `[UPDATE-AFTER-ONI]` markers
and replace with the new numbers.**

---

## STEP 2 — Generate 3 Missing Figures (1 hour)

```bash
python3 run_pipeline.py --only 10
```

This generates: fig09 (SARIMAX PI), fig10 (SHAP beeswarm), fig11 (OLS-XAI).
All other figures are complete.

---

## STEP 3 — Final Manuscript Polish (1 hour)

Open `PV_Leakage_Manuscript_Final.docx` and:

1. Replace all author placeholders `[Author 1]`, `[email]`, etc.
2. Add DOIs to references where known
3. Add actual figure captions (13 captions needed)
4. Verify word count < 10,000 (Energy AI limit)
5. Check journal formatting guidelines

---

## STEP 4 — Submit to Energy AI

**Submission URL:** https://www.editorialmanager.com/eai/

**Files to upload:**
- `PV_Leakage_Manuscript_Final.docx` (main manuscript)
- `PV_Leakage_Supplementary.docx` (Tables S1–S5)
- 13 figure PNG files (300 DPI, listed below)
- Cover letter (from document 24)

**Figures for upload:**
```
fig01_research_framework.png
fig02_leakage_demonstration.png
figNEW_A_sensitivity_heatmap.png
fig03_data_profile.png
fig04_seasonal_climatology.png
fig05_enso_teleconnection.png       ← regenerate after ONI
fig06_stochastic_target_architecture.png
fig07_walkforward_scheme.png
fig08_model_performance.png
fig09_sarimax_pi.png                ← generate in Step 2
fig10_shap_summary.png              ← generate in Step 2
fig11_ols_xai_correspondence.png    ← generate in Step 2
fig12_residual_diagnostics.png
figNEW_B_enso_violin.png            ← regenerate after ONI
```

---

## KEY NUMBERS TO KNOW BY HEART

| Metric | Value | Where it appears |
|---|---|---|
| Leakage lower bound | **2.54×** | Abstract, §4.1 |
| Leakage mean | 3.47× (CV=18.0%) | §4.1, Table 3 |
| XGBoost mean SS | +0.085 | Abstract, §4.3 |
| DM (XGB vs OLS) | p = 0.960 | Abstract, §4.3 |
| GHI_anom p-value | < 0.001 | Abstract, §4.2 |
| SARIMAX PICP | 0.935 | Abstract, §4.3 |
| Winkler Score | 0.386 kWh/m²/day | §4.3, Table 5 |
| KW ENSO test | p > 0.65 | Abstract, §4.4 |

---

## IF REJECTED FROM ENERGY AI

Wait for reviewer comments. Use doc 24 (reviewer response template) —
most responses are pre-written. Typical revision cycle: 6–8 weeks.

If rejected without review → submit to **Solar Energy** (after ONI confirmed).
If rejected after review → submit to **Renewable Energy**.
Last resort → **Energies (MDPI)**, ~60% acceptance, open access.

---

## ALL DELIVERABLE FILES (28 documents produced)

| File | Content |
|---|---|
| **PV_Leakage_Manuscript_Final.docx** | Complete submission manuscript |
| **PV_Leakage_Supplementary.docx** | Tables S1–S5 |
| 19_complete_manuscript.md | Markdown source |
| 25_final_integrated_manuscript.md | Integrated with all corrections |
| 24_cover_letter_and_response.md | Cover letter + reviewer response |
| 26_github_readme.md | Repository README |
| 27_oni_integration_complete.py | ONI integration code |
| 20_peer_review_simulation.md | 5-reviewer simulation |
| 21_final_revision_guide.md | Exact text changes |
| 22_execution_roadmap.md | Implementation steps |
| 28_project_summary.md | Complete journey summary |
| NB01–NB10 | Python pipeline notebooks |

---

**Estimated time from now to journal submission: 1–2 days of focused work.**
**Estimated time to first decision: 6–12 weeks.**
**Estimated acceptance probability (Energy AI): 50–56%.**

