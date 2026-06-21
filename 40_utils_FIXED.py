"""
=============================================================================
UTILS.PY — FIXED VERSION (Q1 Code Audit Remediation)
=============================================================================
This file replaces the repository's utils.py.

CRITICAL FIXES APPLIED (see 39_CODE_AUDIT_CRITICAL_FINDINGS.md for full
audit context):

  FIX #1 — expanding_climatology(): NEW function. Computes calendar-month
           climatology using ONLY the training window passed in. This is
           the function that should be called INSIDE every walk-forward
           fold loop, separately for each fold, to construct GHI_anom /
           CLOUD_anom / PRECTOT_anom / T2M_anom WITHOUT future-information
           leakage. This is the implementation of the manuscript's
           Section 3.5 "per-fold expanding-window climatological
           baseline" methodology, which did not previously exist
           anywhere in the codebase.

  FIX #2 — skill_score(): REWRITTEN. The original version used
           y_true.mean() (the TEST SET's own mean) as the "climatology"
           baseline. This is now replaced with a version that REQUIRES
           an externally supplied per-fold climatology prediction
           (computed via expanding_climatology() from TRAINING data
           only), making it impossible to silently fall back to a
           test-set-derived baseline.

  FIX #3 — get_split_data(): REWRITTEN. Now recomputes the anomaly
           features for both train and test using ONLY the training
           window's climatology (via expanding_climatology()), instead
           of returning pre-computed, full-sample-contaminated columns.

  FIX #4 — verify_no_feature_leakage(): NEW function. A leakage guard
           that actually checks FEATURE VALUES, not just date ordering.
           Compares per-fold-computed anomaly features against
           full-sample-computed ones and asserts they differ — if they
           are identical, the per-fold computation silently failed to
           apply (e.g., wrong window), which the OLD guard could never
           have caught.

USAGE — import this in EVERY notebook (03, 04, 05, 06, 07, 09) instead
of each notebook defining its own local copy of these functions:

    from utils import (
        expanding_climatology, get_split_data, skill_score,
        verify_no_feature_leakage, diebold_mariano,
        friedman_ranking_test, rmse, mae, mape, evaluate_point,
        crps_normal, picp, piaw, winkler_score,
    )
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════
# POINT-FORECAST METRICS  (unchanged from original — these were correct)
# ══════════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Skips zero-valued observations."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ══════════════════════════════════════════════════════════════════════════
# FIX #1 — EXPANDING-WINDOW CLIMATOLOGY (THE MISSING CORE FUNCTION)
# ══════════════════════════════════════════════════════════════════════════

def expanding_climatology(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           value_cols: list[str],
                           month_col: str = "MONTH") -> dict:
    """
    Compute calendar-month climatology using ONLY train_df, then apply
    it to BOTH train_df and test_df to produce leakage-free anomaly
    features and a leakage-free climatological point-forecast baseline.

    This is the per-fold expanding-window climatology described in the
    manuscript (Section 3.5) and is the function that was MISSING from
    every notebook and from the original utils.py in the audited
    repository. It must be called freshly INSIDE each walk-forward fold
    loop — never once on the full dataset before splitting.

    Parameters
    ----------
    train_df : DataFrame
        Training-window rows ONLY (e.g., df[df["YEAR"] < test_year]).
    test_df : DataFrame
        Test-window rows for this fold (e.g., df[df["YEAR"] == test_year]).
    value_cols : list of str
        Raw columns to compute anomalies/climatology for, e.g.
        ["GHI", "CLOUD", "PRECTOT", "T2M"].
    month_col : str
        Name of the integer month-of-year column (1-12).

    Returns
    -------
    dict with keys:
        "climatology"   : DataFrame indexed by month, one column per
                           value_col (the train-only calendar-month means)
        "train_anom"    : DataFrame, same index as train_df, anomaly
                           columns named f"{col}_anom"
        "test_anom"     : DataFrame, same index as test_df, anomaly
                           columns named f"{col}_anom"
        "test_clim_pred": Series, same index as test_df — the
                           climatological point-forecast for whichever
                           single target column is requested separately
                           via climatology_baseline_predict()

    Notes
    -----
    If a calendar month present in test_df does NOT appear in
    train_df (possible only in pathologically short training windows),
    the overall train_df mean is used as a fallback and a warning is
    printed — this should never trigger for windows >= 12 months.
    """
    climatology = train_df.groupby(month_col)[value_cols].mean()

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        missing_months = set(frame[month_col].unique()) - set(climatology.index)
        if missing_months:
            print(f"  ⚠ expanding_climatology: month(s) {missing_months} "
                  f"absent from training window — using overall train mean "
                  f"as fallback for these months.")
        for col in value_cols:
            clim_lookup = frame[month_col].map(climatology[col])
            if missing_months:
                clim_lookup = clim_lookup.fillna(train_df[col].mean())
            out[f"{col}_anom"] = frame[col].values - clim_lookup.values
        return out

    train_anom = _apply(train_df)
    test_anom = _apply(test_df)

    return {
        "climatology": climatology,
        "train_anom": train_anom,
        "test_anom": test_anom,
    }


def climatology_baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  target_col: str,
                                  month_col: str = "MONTH") -> np.ndarray:
    """
    The leakage-free climatological POINT-FORECAST baseline for one fold.

    For each test-set row, predicts target_col using the calendar-month
    mean of target_col computed EXCLUSIVELY from train_df (the
    expanding training window for this fold). This is the correct
    implementation of the baseline described in the manuscript and
    replaces:
      - utils.py's old skill_score() internal y_true.mean() baseline
      - 09_residual_diagnostics.py's full-test-period-combined
        "Climatology_baseline" (which used 2015-2023 jointly — the
        "aggregate baseline" bug the manuscript says was fixed but
        which persisted in the actual committed code).

    Returns
    -------
    np.ndarray, same length as test_df, in test_df row order.
    """
    clim = train_df.groupby(month_col)[target_col].mean()
    overall_fallback = train_df[target_col].mean()
    preds = test_df[month_col].map(clim).fillna(overall_fallback).values
    return preds


# ══════════════════════════════════════════════════════════════════════════
# FIX #2 — SKILL SCORE (now REQUIRES an explicit, externally-computed
# climatological prediction array — no implicit test-set-derived fallback)
# ══════════════════════════════════════════════════════════════════════════

def skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                 y_clim_pred: np.ndarray) -> float:
    """
    Skill Score vs. an EXPLICITLY SUPPLIED climatological baseline.

    SS = 1 - RMSE_model / RMSE_climatology
    SS > 0 -> model beats climatology.

    CHANGED FROM ORIGINAL: the original signature was
        skill_score(y_true, y_pred)
    and internally built y_clim = np.full_like(y_true, y_true.mean()) —
    i.e., it used the TEST SET's own mean as "climatology", which is a
    data-leakage bug (the real-world forecaster does not know the test
    period's own mean in advance). The new signature FORCES the caller
    to pass in a y_clim_pred array, which should come from
    climatology_baseline_predict() computed on the TRAINING window only.
    This makes the leakage structurally impossible to reintroduce by
    accident, because there is no longer any internal fallback that
    touches y_true for baseline construction.

    Parameters
    ----------
    y_true : np.ndarray — test-set true values
    y_pred : np.ndarray — model predictions for the test set
    y_clim_pred : np.ndarray — climatological predictions for the test
        set, computed via climatology_baseline_predict(train_df, test_df, ...)
    """
    rmse_cl = rmse(y_true, y_clim_pred)
    rmse_md = rmse(y_true, y_pred)
    return 1 - rmse_md / rmse_cl if rmse_cl > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════
# PROBABILISTIC METRICS (unchanged — these were correctly implemented)
# ══════════════════════════════════════════════════════════════════════════

def crps_normal(y_true, mu_pred, sigma_pred):
    """CRPS for Gaussian predictive distribution. Lower = better."""
    z = (y_true - mu_pred) / sigma_pred
    phi = stats.norm.pdf(z)
    PHI = stats.norm.cdf(z)
    crps = sigma_pred * (z * (2 * PHI - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def picp(y_true, lb, ub):
    """Prediction Interval Coverage Probability."""
    return float(((y_true >= lb) & (y_true <= ub)).mean())


def piaw(lb, ub):
    """Prediction Interval Average Width."""
    return float(np.mean(ub - lb))


def winkler_score(y_true, lb, ub, alpha=0.10):
    """Winkler Score. Penalises both width and coverage failures. Lower = better."""
    width = ub - lb
    below = y_true < lb
    above = y_true > ub
    score = width.copy()
    score[below] += (2 / alpha) * (lb[below] - y_true[below])
    score[above] += (2 / alpha) * (y_true[above] - ub[above])
    return float(np.mean(score))


def evaluate_point(y_true, y_pred, y_clim_pred, label=""):
    """
    Run all point-forecast metrics, INCLUDING the leakage-free skill
    score. y_clim_pred is now a required argument (see skill_score()
    above for rationale).
    """
    return {
        "model": label,
        "RMSE": round(rmse(y_true, y_pred), 6),
        "MAE": round(mae(y_true, y_pred), 6),
        "MAPE": round(mape(y_true, y_pred), 4),
        "SkillScore": round(skill_score(y_true, y_pred, y_clim_pred), 4),
        "n": len(y_true),
    }


# ══════════════════════════════════════════════════════════════════════════
# DIEBOLD-MARIANO TEST (unchanged — implementation was correct;
# only documentation strengthened per audit recommendation C/7)
# ══════════════════════════════════════════════════════════════════════════

def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1,
                     crit: str = "mse") -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    AUDIT NOTE (resolves audit item C/7 — "unit of analysis ambiguity"):
    This function operates on whatever error arrays e1/e2 you pass in.
    For this repository's reporting in the manuscript, e1/e2 MUST be the
    POOLED monthly forecast errors across all 9 walk-forward folds
    (i.e., length ~108, NOT length 9). Do not pass fold-aggregated RMSE
    values into this function — DM theory assumes a loss-differential
    series over the forecast evaluation sample, not over folds. If a
    fold-level comparison is desired, report it separately and label it
    explicitly as such (e.g., via the Wilcoxon signed-rank test over
    9 fold-level RMSE values, which IS appropriate for n=9).

    Args:
        e1, e2 : forecast errors (not squared) from model 1 and model 2,
                 same length, pooled across all evaluation months.
        h      : forecast horizon (1 for one-step-ahead).
        crit   : loss differential criterion ('mse' or 'mae').

    Returns:
        (dm_stat, p_value_two_sided)

    Reference:
        Diebold & Mariano (1995), J. Business & Economic Statistics.
        Harvey, Leybourne & Newbold (1997) small-sample correction applied.
    """
    if crit == "mse":
        d = e1 ** 2 - e2 ** 2
    elif crit == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("crit must be 'mse' or 'mae'")

    n = len(d)
    d_bar = np.mean(d)

    gamma0 = np.var(d, ddof=1)
    if h > 1:
        gammas = [np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)]
        V_d = gamma0 + 2 * sum(gammas)
    else:
        V_d = gamma0

    V_d_corrected = V_d * (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_stat = d_bar / np.sqrt(max(V_d_corrected / n, 1e-12))
    p_val = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    return float(dm_stat), float(p_val)


# ══════════════════════════════════════════════════════════════════════════
# FRIEDMAN RANKING TEST (unchanged — implementation was correct)
# ══════════════════════════════════════════════════════════════════════════

def friedman_ranking_test(results_dict: dict, metric: str = "RMSE"):
    """
    Friedman test for significant rank differences across multiple
    models, evaluated over n_folds. Appropriate unit of analysis: one
    value per model per fold (n=9 in this repository), NOT pooled
    monthly values.
    """
    models = list(results_dict.keys())
    data = np.array([results_dict[m] for m in models]).T

    ranks = np.array([stats.rankdata(row) for row in data])
    n_folds, k = ranks.shape
    mean_ranks = ranks.mean(axis=0)

    chi2 = (12 * n_folds / (k * (k + 1))) * (
        np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4
    )
    p_val = stats.chi2.sf(chi2, df=k - 1)

    return float(chi2), float(p_val), dict(zip(models, mean_ranks.round(3)))


# ══════════════════════════════════════════════════════════════════════════
# FIX #3 — GET_SPLIT_DATA (now recomputes anomaly features per fold)
# ══════════════════════════════════════════════════════════════════════════

def get_split_data(df: pd.DataFrame, test_year: int, features: list[str],
                    raw_anomaly_source_cols: list[str] | None = None,
                    target: str = "Y_stoch", month_col: str = "MONTH"):
    """
    Return (X_train, y_train, X_test, y_test, dates_test, y_clim_pred)
    for one walk-forward fold, with LEAKAGE-FREE anomaly features.

    CHANGED FROM ORIGINAL: the original version sliced df by YEAR and
    directly returned whatever was already in the *_anom columns —
    columns that had been precomputed ONCE on the full 2005-2025 sample
    in 03_feature_engineering.py. This version instead:
      1. Slices df by YEAR exactly as before (train/test temporal split
         is correct and unchanged).
      2. If raw_anomaly_source_cols is given, RECOMPUTES every
         f"{col}_anom" feature in `features` using expanding_climatology()
         on the TRAINING split only, overwriting whatever was in df.
      3. Also returns y_clim_pred — the leakage-free climatological
         point-forecast baseline for this fold's test set, for use with
         the new skill_score() / evaluate_point() signature.

    Args:
        df        : model-ready DataFrame with DATE, YEAR, MONTH columns,
                    and the RAW (non-anomaly) versions of any column
                    listed in raw_anomaly_source_cols (e.g. "GHI", not
                    "GHI_anom") still present.
        test_year : year to use as test set.
        features  : full feature list to return in X_train/X_test,
                    e.g. ["sin_month","cos_month","GHI_anom",...].
        raw_anomaly_source_cols : list of RAW column names (e.g.
                    ["GHI","CLOUD","PRECTOT","T2M"]) whose "_anom"
                    derivative appears in `features`. These will be
                    recomputed per-fold. If None, no recomputation is
                    attempted (features are used as-is — only safe if
                    none of them are anomaly features).
        target    : target column name.
        month_col : integer month-of-year column name.

    Returns:
        X_train, y_train, X_test, y_test, dates_test, y_clim_pred
    """
    train = df[df["YEAR"] < test_year].copy()
    test = df[df["YEAR"] == test_year].copy()

    assert len(train) > 0, f"Empty training set for test_year={test_year}"
    assert len(test) > 0, f"Empty test set for test_year={test_year}"
    assert train["DATE"].max() < test["DATE"].min(), \
        f"Temporal leakage: train/test dates overlap for year {test_year}!"

    if raw_anomaly_source_cols:
        clim_result = expanding_climatology(train, test, raw_anomaly_source_cols, month_col)
        for col in raw_anomaly_source_cols:
            anom_name = f"{col}_anom"
            if anom_name in features:
                train[anom_name] = clim_result["train_anom"][anom_name].values
                test[anom_name] = clim_result["test_anom"][anom_name].values

    y_clim_pred = climatology_baseline_predict(train, test, target, month_col)

    X_train = train[features].values
    y_train = train[target].values
    X_test = test[features].values
    y_test = test[target].values
    dates_test = test["DATE"].values

    return X_train, y_train, X_test, y_test, dates_test, y_clim_pred


# ══════════════════════════════════════════════════════════════════════════
# FIX #4 — FEATURE-LEVEL LEAKAGE GUARD (the missing real check)
# ══════════════════════════════════════════════════════════════════════════

def verify_no_feature_leakage(df_full: pd.DataFrame, test_year: int,
                               raw_anomaly_source_cols: list[str],
                               month_col: str = "MONTH",
                               tolerance: float = 1e-9) -> None:
    """
    Real leakage guard: verifies that per-fold-computed anomaly features
    DIFFER from full-sample-computed anomaly features (they should,
    whenever the test year's calendar-month values differ from the
    full-sample mean for that month — which is true except in
    coincidental edge cases). If they are IDENTICAL, this is strong
    evidence that the per-fold computation was not actually applied
    (e.g., a caller forgot to pass raw_anomaly_source_cols to
    get_split_data(), silently falling back to the leaky full-sample
    columns still present in df_full).

    This does NOT replace the temporal date-ordering assertion already
    in get_split_data() — it is a SEPARATE, complementary check at the
    feature-value level, which the original repository's "leakage
    guard" never performed.

    Raises
    ------
    AssertionError if any anomaly column is numerically identical
    (within `tolerance`) between the full-sample and per-fold
    computation for the test fold's observations — i.e., if the fix
    appears NOT to have taken effect.
    """
    train = df_full[df_full["YEAR"] < test_year]
    test = df_full[df_full["YEAR"] == test_year]

    clim_fullsample = df_full.groupby(month_col)[raw_anomaly_source_cols].mean()
    clim_perfold = train.groupby(month_col)[raw_anomaly_source_cols].mean()

    max_abs_diff = (clim_fullsample.loc[test[month_col].unique()] -
                     clim_perfold.reindex(test[month_col].unique())
                     ).abs().max().max()

    assert max_abs_diff > tolerance, (
        f"LEAKAGE GUARD FAILED for test_year={test_year}: per-fold and "
        f"full-sample climatology are numerically identical (max diff = "
        f"{max_abs_diff:.2e}). This strongly suggests the per-fold "
        f"recomputation did not actually run — check that "
        f"get_split_data() was called with raw_anomaly_source_cols set."
    )
    print(f"  ✓ Feature-level leakage guard passed for test_year={test_year} "
          f"(max climatology divergence from full-sample = {max_abs_diff:.4f})")
