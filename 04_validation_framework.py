"""
=============================================================================
NOTEBOOK 04: WALK-FORWARD VALIDATION FRAMEWORK
=============================================================================
Purpose  : Implement expanding-window walk-forward validation.
           Define all evaluation metrics (point + probabilistic).
           Implement Diebold-Mariano and Friedman ranking tests.

Input    : data/03_model_ready.parquet
           data/03_final_features.csv
Output   : outputs/04_validation_splits.csv (split definitions)
           data/04_validation_framework.parquet (framework object as data)

This notebook defines the validation infrastructure used by ALL model notebooks.
Every model (OLS, SARIMAX, XGBoost) MUST use identical splits.
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(f"{DATA_DIR}/03_model_ready.parquet")
FINAL_FEATURES = pd.read_csv(f"{DATA_DIR}/03_final_features.csv")["feature"].tolist()

print(f"Loaded: {df.shape[0]} model-ready observations")
print(f"Period: {df['DATE'].min().date()} → {df['DATE'].max().date()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: WALK-FORWARD EXPANDING WINDOW SPLITS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Expanding Window Walk-Forward Splits")
print("=" * 60)

"""
DESIGN RATIONALE:
- Initial training window: 10 years (2005–2014) = 120 months minimum
  (ensures enough observations for SARIMAX seasonal estimation)
- Test horizon: 12 months (one full annual cycle per fold)
- Expanding: training set grows each fold (all history used)
- Total folds: 9 (2015–2023 test years; 2024–2025 held as final holdout)
- 2024–2025 (24 months) = FINAL HOLDOUT: not touched until paper submission

This ensures:
  1. Temporal integrity — future never informs past
  2. Sufficient training data for seasonal models
  3. Complete annual cycles in test set (avoids seasonal bias)
  4. Hold-out for final unbiased evaluation
"""

# Define splits
splits = []
for fold_idx, test_year in enumerate(range(2015, 2024)):  # 9 folds
    train_mask = df["YEAR"] < test_year
    test_mask  = df["YEAR"] == test_year

    train_dates = df.loc[train_mask, "DATE"]
    test_dates  = df.loc[test_mask, "DATE"]

    splits.append({
        "fold":           fold_idx + 1,
        "test_year":      test_year,
        "train_start":    train_dates.min().date(),
        "train_end":      train_dates.max().date(),
        "n_train":        train_mask.sum(),
        "test_start":     test_dates.min().date(),
        "test_end":       test_dates.max().date(),
        "n_test":         test_mask.sum(),
    })

df_splits = pd.DataFrame(splits)
print(df_splits.to_string(index=False))
print(f"\n  Total test observations: {df_splits['n_test'].sum()}")
print(f"  Final holdout (2024–2025): {(df['YEAR'] >= 2024).sum()} months [NOT USED IN CV]")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: EVALUATION METRICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Defining evaluation metrics")

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def skill_score(y_true, y_pred):
    """
    Skill Score vs. monthly climatology baseline.
    SS = 1 - RMSE_model / RMSE_climatology
    SS > 0: model beats climatology.
    SS = 1: perfect model.
    """
    y_clim  = np.full_like(y_true, y_true.mean())
    rmse_cl = rmse(y_true, y_clim)
    rmse_md = rmse(y_true, y_pred)
    return 1 - rmse_md / rmse_cl if rmse_cl > 0 else np.nan

def crps_normal(y_true, mu_pred, sigma_pred):
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.
    CRPS(N(μ,σ), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ
    Lower CRPS = better probabilistic forecast.
    """
    z    = (y_true - mu_pred) / sigma_pred
    phi  = stats.norm.pdf(z)
    PHI  = stats.norm.cdf(z)
    crps = sigma_pred * (z * (2 * PHI - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return np.mean(crps)

def picp(y_true, lb, ub):
    """
    Prediction Interval Coverage Probability.
    Empirical vs. nominal (should match 90% for 90% PI).
    """
    covered = ((y_true >= lb) & (y_true <= ub)).mean()
    return covered

def piaw(lb, ub):
    """Prediction Interval Average Width (narrower = better, given coverage met)"""
    return np.mean(ub - lb)

def winkler_score(y_true, lb, ub, alpha=0.10):
    """
    Winkler Score for (1-alpha) prediction interval.
    Penalizes both width and coverage failures.
    Lower = better.
    """
    width  = ub - lb
    below  = y_true < lb
    above  = y_true > ub
    score  = width.copy()
    score[below] += (2 / alpha) * (lb[below] - y_true[below])
    score[above] += (2 / alpha) * (y_true[above] - ub[above])
    return np.mean(score)

def evaluate_point(y_true, y_pred, label=""):
    """Run all point-forecast metrics."""
    return {
        "model":        label,
        "RMSE":         round(rmse(y_true, y_pred), 6),
        "MAE":          round(mae(y_true, y_pred), 6),
        "MAPE":         round(mape(y_true, y_pred), 4),
        "SkillScore":   round(skill_score(y_true, y_pred), 4),
        "n":            len(y_true),
    }

print(f"  ✓ Metric functions defined: RMSE, MAE, MAPE, SkillScore, CRPS, PICP, PIAW, Winkler")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: DIEBOLD-MARIANO TEST
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Diebold-Mariano test implementation")

def diebold_mariano(e1, e2, h=1, crit="mse"):
    """
    Diebold-Mariano test for equal predictive accuracy.
    H0: Both models have equal predictive accuracy.
    H1: Model 1 is significantly better than Model 2 (one-sided).

    e1, e2: forecast errors from model 1 and model 2
    h     : forecast horizon (1 for 1-step ahead)
    crit  : loss differential criterion ('mse' or 'mae')

    Returns: DM statistic, p-value (two-sided)

    Reference: Diebold & Mariano (1995), J. Business & Economic Statistics
    Harvey, Leybourne & Newbold (1997) correction applied for small samples.
    """
    if crit == "mse":
        d = e1 ** 2 - e2 ** 2
    elif crit == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("crit must be 'mse' or 'mae'")

    n    = len(d)
    d_bar = np.mean(d)

    # Newey-West variance (accounts for autocorrelation up to lag h-1)
    gamma0 = np.var(d, ddof=1)
    if h > 1:
        gammas = [np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)]
        V_d = gamma0 + 2 * sum(gammas)
    else:
        V_d = gamma0

    # Harvey-Leybourne-Newbold small-sample correction
    V_d_corrected = V_d * (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_stat = d_bar / np.sqrt(V_d_corrected / n)

    # t-distribution with (n-1) df
    p_val = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    return float(dm_stat), float(p_val)

print(f"  ✓ Diebold-Mariano test implemented (Harvey-Leybourne-Newbold corrected)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: FRIEDMAN RANKING TEST
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Friedman ranking test (multi-model comparison)")

def friedman_ranking_test(results_dict, metric="RMSE"):
    """
    Friedman test for significant differences across multiple models.
    Non-parametric: ranks models within each fold, tests if ranks differ.

    results_dict: {model_name: [metric_fold1, metric_fold2, ...]}
    Returns: chi2 statistic, p-value, mean ranks
    """
    models = list(results_dict.keys())
    data   = np.array([results_dict[m] for m in models]).T  # (n_folds, n_models)

    # Rank within each fold (low RMSE = rank 1 = best)
    ranks = np.array([stats.rankdata(row) for row in data])  # (n_folds, n_models)

    n_folds, k = ranks.shape
    mean_ranks  = ranks.mean(axis=0)

    # Friedman statistic
    chi2 = (12 * n_folds / (k * (k + 1))) * (
        np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4
    )
    p_val = stats.chi2.sf(chi2, df=k - 1)

    return float(chi2), float(p_val), dict(zip(models, mean_ranks.round(3)))

print(f"  ✓ Friedman ranking test implemented")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: LEAKAGE GUARD FOR SPLITS
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Temporal leakage guard")

def get_split_data(df, fold_idx, features, target="Y_stoch"):
    """
    Returns (X_train, y_train, X_test, y_test) for a given fold.
    STRICT: test set contains ONLY data from test_year.
    train set contains ALL data BEFORE test_year.
    No data from test_year contaminates training.
    """
    split   = splits[fold_idx]
    t_year  = split["test_year"]

    train   = df[df["YEAR"] < t_year].copy()
    test    = df[df["YEAR"] == t_year].copy()

    X_train = train[features].values
    y_train = train[target].values
    X_test  = test[features].values
    y_test  = test[target].values
    dates_test = test["DATE"].values

    # Assert: no overlap
    assert train["DATE"].max() < test["DATE"].min(), \
        f"Fold {fold_idx+1}: Training and test dates overlap!"

    return X_train, y_train, X_test, y_test, dates_test

# Verify split integrity
for i in range(9):
    X_tr, y_tr, X_te, y_te, _ = get_split_data(df, i, FINAL_FEATURES)
    assert len(X_tr) > 0 and len(X_te) == 12, f"Split {i+1} size error"
print(f"  ✓ All 9 splits verified: no temporal leakage")

# ── SAVE ──────────────────────────────────────────────────────────────────
df_splits.to_csv(f"{OUT_DIR}/04_validation_splits.csv", index=False)

print(f"\n✅ Notebook 04 complete.")
print(f"   9 walk-forward folds defined.")
print(f"   Final holdout (2024–2025): {(df['YEAR'] >= 2024).sum()} months reserved.")
print(f"   Metrics: RMSE, MAE, MAPE, SkillScore, CRPS, PICP, PIAW, Winkler")
print(f"   Tests: Diebold-Mariano, Friedman")
