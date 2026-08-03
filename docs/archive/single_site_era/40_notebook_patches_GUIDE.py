"""
=============================================================================
PATCH FILE — Concrete fixes for notebooks 03, 04, 05, 06, 07, 09
=============================================================================
This is a copy-paste-ready patch guide, not a standalone runnable script.
Apply each block to the corresponding notebook file. Verified against the
fixed utils.py (40_utils_FIXED.py) via functional tests (see audit report
39_CODE_AUDIT_CRITICAL_FINDINGS.md, Section 7).
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════
# PATCH FOR: notebooks/03_feature_engineering.py
# ═══════════════════════════════════════════════════════════════════════
"""
DELETE this block (the original "STEP 2: CLIMATOLOGICAL ANOMALIES" — this
is the source of the critical leakage finding #1):

    # ── DELETE FROM HERE ──────────────────────────────────────────────
    for col in ["GHI", "CLOUD", "PRECTOT", "T2M"]:
        climatology = df.groupby("MONTH")[col].mean().rename(f"{col}_clim")
        df = df.merge(climatology, on="MONTH", how="left")
        df[f"{col}_anom"] = df[col] - df[f"{col}_clim"]
        df.drop(columns=[f"{col}_clim"], inplace=True)
    # ── DELETE TO HERE ────────────────────────────────────────────────

REPLACE WITH:

    # Anomaly features are NOT computed here anymore. They are computed
    # PER WALK-FORWARD FOLD, using only that fold's training window, via
    # utils.expanding_climatology(). This eliminates the data leakage
    # confirmed in the Q1 code audit (see 39_CODE_AUDIT_CRITICAL_FINDINGS.md,
    # finding #1): the previous implementation computed GHI_anom,
    # CLOUD_anom, PRECTOT_anom, and T2M_anom using the FULL 2005-2025
    # sample's calendar-month means, contaminating every fold's features
    # (including training folds) with information from years that would
    # not yet be observed at that fold's forecast origin.
    #
    # This notebook still exports the RAW columns (GHI, CLOUD, PRECTOT,
    # T2M) needed downstream; anomaly columns are added fresh inside each
    # fold's loop in notebooks 05, 06, 07, and 09 via:
    #
    #     from utils import get_split_data
    #     X_tr, y_tr, X_te, y_te, dates_te, y_clim = get_split_data(
    #         df, test_year, features=FINAL_FEATURES,
    #         raw_anomaly_source_cols=["GHI", "CLOUD", "PRECTOT", "T2M"],
    #         target="Y_stoch"
    #     )
    print("  NOTE: Anomaly features (*_anom) are intentionally NOT computed "
          "in this notebook. They are computed per-fold in the modeling "
          "notebooks (05/06/07/09) via utils.get_split_data() to prevent "
          "the climatology leakage identified in the Q1 code audit.")

IMPORTANT DOWNSTREAM CONSEQUENCE:
  - df_model = df.dropna(subset=FINAL_FEATURES + ["Y_stoch"]) (Step 9 VIF
    analysis) will FAIL because *_anom columns no longer exist in df at
    this point. For the VIF / correlation matrix report (a DESCRIPTIVE,
    full-sample diagnostic — not a predictive evaluation step, so full-
    sample computation is legitimate there), compute a SEPARATE, clearly
    labeled full-sample anomaly version FOR REPORTING PURPOSES ONLY:

    df_report = df.copy()
    for col in ["GHI", "CLOUD", "PRECTOT", "T2M"]:
        df_report[f"{col}_anom"] = df_report[col] - df_report.groupby("MONTH")[col].transform("mean")
    # Use df_report ONLY for VIF_report.csv / correlation_matrix.csv.
    # Never use df_report's *_anom columns for any walk-forward model fit.
"""

# ═══════════════════════════════════════════════════════════════════════
# PATCH FOR: notebooks/05_ols_hc3_model.py  (Part D: Walk-Forward Evaluation)
# ═══════════════════════════════════════════════════════════════════════
"""
DELETE:

    # ── DELETE FROM HERE ──────────────────────────────────────────────
    for fold_idx, test_year in enumerate(range(2015, 2024)):
        train = df[df["YEAR"] < test_year]
        test  = df[df["YEAR"] == test_year]

        X_tr = sm.add_constant(train[FINAL_FEATURES], has_constant="add")
        y_tr = train[TARGET]
        X_te = sm.add_constant(test[FINAL_FEATURES], has_constant="add")
        y_te = test[TARGET].values

        fold_ols = sm.OLS(y_tr, X_tr).fit(cov_type="HC3")
        y_pred   = fold_ols.predict(X_te)

        fold_rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
        fold_mae  = np.mean(np.abs(y_te - y_pred))
        fold_ss   = 1 - fold_rmse / np.sqrt(np.mean((y_te - y_te.mean()) ** 2))
    # ── DELETE TO HERE ────────────────────────────────────────────────

REPLACE WITH:

    from utils import get_split_data, skill_score, verify_no_feature_leakage

    RAW_ANOM_COLS = ["GHI", "CLOUD", "PRECTOT", "T2M"]

    for fold_idx, test_year in enumerate(range(2015, 2024)):
        # Real leakage guard — fails loudly if per-fold recomputation
        # is not actually taking effect.
        verify_no_feature_leakage(df, test_year, RAW_ANOM_COLS)

        X_tr_arr, y_tr_arr, X_te_arr, y_te, dates_te, y_clim_pred = get_split_data(
            df, test_year, features=FINAL_FEATURES,
            raw_anomaly_source_cols=RAW_ANOM_COLS, target=TARGET
        )

        X_tr = sm.add_constant(pd.DataFrame(X_tr_arr, columns=FINAL_FEATURES),
                                has_constant="add")
        X_te = sm.add_constant(pd.DataFrame(X_te_arr, columns=FINAL_FEATURES),
                                has_constant="add")

        fold_ols = sm.OLS(y_tr_arr, X_tr).fit(cov_type="HC3")
        y_pred   = fold_ols.predict(X_te).values

        fold_rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
        fold_mae  = np.mean(np.abs(y_te - y_pred))
        fold_ss   = skill_score(y_te, y_pred, y_clim_pred)   # LEAKAGE-FREE

EXPECTED IMPACT ON RESULTS: fold_rmse, fold_ss, and the OLS coefficient
on GHI_anom (and other anomaly features) computed in Part A (full-sample
fit) may change modestly after this fix. This is EXPECTED and CORRECT —
any change reflects removal of look-ahead bias, not a new error. Re-run
the full pipeline and update Tables 5/6 with the new numbers before
resubmission.
"""

# ═══════════════════════════════════════════════════════════════════════
# PATCH FOR: notebooks/06_sarimax_climate_model.py and
#            notebooks/07_xgboost_model.py  (walk-forward loops)
# ═══════════════════════════════════════════════════════════════════════
"""
Both notebooks have the identical bug pattern:

    fold_ss = 1 - fold_rmse / (np.std(y_te) if np.std(y_te) > 0 else 1)

DELETE this line in both files. REPLACE with the same pattern as the
NB05 patch above: call get_split_data(..., raw_anomaly_source_cols=...)
at the top of each fold's loop body to obtain y_clim_pred, then:

    fold_ss = skill_score(y_te, y_pred, y_clim_pred)

For 06_sarimax_climate_model.py specifically: SARIMAX uses
df.set_index("DATE") rather than a plain DataFrame slice — adapt by
calling get_split_data() on the UN-indexed df (before .set_index) to
obtain X_tr/X_te/y_clim_pred, then separately constructing the
DATE-indexed exog/endog Series needed by statsmodels' SARIMAX class from
the same train/test row masks. Do not let SARIMAX's own data handling
bypass the leakage-free anomaly recomputation.

For 07_xgboost_model.py specifically: apply the SAME get_split_data()
call inside the Part B walk-forward loop. The Part A hyperparameter
search (fold-1 only, inner_train/inner_val) is methodologically correct
as-is and does NOT need this patch, since GHI_anom etc. recomputed for
just the 2005-2014 window vs the FULL 2005-2025 window would differ from
each other but the hyperparameter search doesn't use Y_stoch's
relationship to climatology directly — however, for full consistency,
it is still RECOMMENDED to recompute anomalies for inner_train/inner_val
using only inner_train (i.e., apply the same fix there too).
"""

# ═══════════════════════════════════════════════════════════════════════
# PATCH FOR: notebooks/09_residual_diagnostics.py (Part A)
# ═══════════════════════════════════════════════════════════════════════
"""
This is the SECOND critical fix — the "Climatology_baseline" row in the
final aggregate comparison table currently uses the full 2015-2023
test-period-combined climatology (the exact "aggregate baseline" bug the
manuscript says was corrected).

DELETE:

    # ── DELETE FROM HERE ──────────────────────────────────────────────
    y_clim = np.tile(df_cmp.groupby(df_cmp["DATE"].dt.month)["y_true"].transform("mean").values, 1)
    y_clim = df_cmp.groupby(df_cmp["DATE"].dt.month)["y_true"].transform("mean").values
    perf_rows.append(compute_metrics(y_true, y_clim, "Climatology_baseline"))
    # ── DELETE TO HERE ────────────────────────────────────────────────

REPLACE WITH:

    # Build the per-fold expanding-window climatological baseline by
    # concatenating each fold's OWN train-only climatology prediction —
    # this is the ONLY correct way to represent "the climatology
    # baseline" in an aggregate table, because a single 2015-2023-wide
    # climatology (the deleted code above) uses future test years to
    # predict earlier test years, which is exactly the bug the
    # manuscript's Methods Section 3.5 says was identified and fixed.
    from utils import climatology_baseline_predict

    df_meta_full = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
    y_clim_parts = []
    for test_year in range(2015, 2024):
        train_y = df_meta_full[df_meta_full["YEAR"] < test_year]
        test_y  = df_meta_full[df_meta_full["YEAR"] == test_year]
        y_clim_parts.append(
            climatology_baseline_predict(train_y, test_y, "Y_stoch")
        )
    y_clim = np.concatenate(y_clim_parts)
    assert len(y_clim) == len(y_true), \
        "Climatology baseline length mismatch — check fold alignment"
    perf_rows.append(compute_metrics(y_true, y_clim, "Climatology_baseline"))

EXPECTED IMPACT: the "Climatology_baseline" RMSE row in
outputs/09_model_comparison_table.csv will likely INCREASE relative to
the old (leaky) value, because the corrected baseline no longer benefits
from seeing the full 9-year test period when predicting each individual
year. This will likely make the models' Skill Scores relative to this
corrected baseline LARGER (more positive), since the corrected baseline
is a fairer, harder-to-game comparator. Re-run and report the new
numbers — do not keep the old Table 6 values once this fix is applied.
"""

print(__doc__)
print("\nThis file is a patch GUIDE, not an executable script.")
print("Apply each documented block to the corresponding notebook file,")
print("then re-run: python run_pipeline.py --from 3")
print("\nAfter re-running, EVERY table/figure depending on GHI_anom,")
print("CLOUD_anom, PRECTOT_anom, T2M_anom, or any SkillScore/baseline")
print("value must be regenerated and re-checked against the manuscript.")
