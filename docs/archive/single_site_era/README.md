# Archive: Single-Site (Bontang-Only) Era

This folder preserves working documents from the project's earlier
single-site phase, before the multi-site expansion to 10 locations
(see `docs/manuscript_10sites/MANUSCRIPT_FULL_10SITES.md` for the
current, active manuscript).

## Why These Files Are Archived, Not Deleted

These documents record the genuine research and engineering history of
this project — including the original Q1 code audit that first
identified and fixed the deterministic-leakage bugs later confirmed
across all 10 sites. They are kept for provenance and reproducibility
of the project's development history, but **none of their specific
numeric claims (R², leakage ratios, coefficients) should be cited** —
those are all Bontang-single-site values, since superseded by the
10-site results in `results/combined/`.

## Contents

| File | Original Purpose (single-site era) |
|---|---|
| `24_cover_letter_and_response.md` | Draft cover letter / reviewer response, single-site submission |
| `29_NEXT_STEPS_ACTION_CARD.md` | Working task list, single-site phase |
| `30_major_revision_complete.md` | Simulated major-revision review, single-site manuscript |
| `31_revision_changelog.md` | Changelog, single-site manuscript revisions |
| `32_final_project_summary.md` | Project summary, single-site phase |
| `39_CODE_AUDIT_CRITICAL_FINDINGS_root.md` / `_docs.md` | Original Q1 code audit (leakage bugs found + fixed) — historically important, still an accurate account of *what was fixed and why* |
| `40_notebook_patches_GUIDE.py` | Patch guide accompanying the code audit above |
| `42_q1_leakage_fix.patch` | Git patch implementing the leakage fixes |
| `43_HOW_TO_APPLY_AND_PUSH_root.md` / `_docs.md` | Instructions for applying the above patch (obsolete — already merged) |
| `44_REAL_DATA_RESULTS_SUMMARY_root.md` / `_docs.md` | Single-site (Bontang) real-data validation after the leakage fix |
| `README_DOCS_stale_index.md` | Old docs/ index, referenced a branch (`fix/q1-leakage-audit`) that has since been merged and superseded |
| `graphical_abstract_bontang_only.svg` | 4-panel graphical abstract with single-site numbers (R²=0.226 etc.) — superseded by `results/combined/figures/` |

## What Superseded These Files

| Old (single-site) | New (10-site, current) |
|---|---|
| Leakage ratio 2.5x (Bontang, 1 site) | 1.72x–3.43x range (10 sites), see `results/combined/TABLE2_leakage_all_sites.csv` |
| Single-site OLS coefficient | Meta-analysis beta=+0.102, I2=0.0%, see `TABLE3_ols_meta_analysis.csv` |
| Single-site graphical abstract | `results/combined/figures/` (12 figures) |
| This archive's manuscript drafts | `docs/manuscript_10sites/MANUSCRIPT_FULL_10SITES.md` |
