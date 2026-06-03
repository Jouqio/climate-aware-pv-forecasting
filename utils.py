"""
=============================================================================
UTILS.PY — Shared utility functions
=============================================================================
Fungsi-fungsi ini sebelumnya didefinisikan ULANG di NB04 dan NB09.
Satukan di sini agar konsisten dan mudah diuji secara independen.

Import di setiap notebook:
    from utils import (
        diebold_mariano, friedman_ranking_test,
        rmse, mae, mape, skill_score,
        crps_normal, picp, piaw, winkler_score,
        evaluate_point, get_split_data,
    )
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════
# POINT-FORECAST METRICS.
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


def skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Skill Score vs monthly climatology baseline.
    SS = 1 - RMSE_model / RMSE_climatology
    SS > 0 → model beats climatology.
    SS = 1 → perfect model.
    """
    y_clim  = np.full_like(y_true, y_true.mean())
    rmse_cl = rmse(y_true, y_clim)
    rmse_md = rmse(y_true, y_pred)
    return float(1 - rmse_md / rmse_cl) if rmse_cl > 0 else float("nan")


def evaluate_point(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """Run all point-forecast metrics and return as dict."""
    return {
        "model":       label,
        "RMSE":        round(rmse(y_true, y_pred),        6),
        "MAE":         round(mae(y_true, y_pred),         6),
        "MAPE_%":      round(mape(y_true, y_pred),        4),
        "SkillScore":  round(skill_score(y_true, y_pred), 4),
        "n":           int(len(y_true)),
    }


# ══════════════════════════════════════════════════════════════════════════
# PROBABILISTIC METRICS
# ══════════════════════════════════════════════════════════════════════════

def crps_normal(
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.
    CRPS(N(μ,σ), y) = σ · [z(2Φ(z)−1) + 2φ(z) − 1/√π]
    where z = (y − μ) / σ
    Lower CRPS = better probabilistic forecast.

    Reference: Gneiting & Raftery (2007), J. American Statistical Association.
    """
    z    = (y_true - mu_pred) / sigma_pred
    phi  = stats.norm.pdf(z)
    PHI  = stats.norm.cdf(z)
    crps = sigma_pred * (z * (2 * PHI - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def picp(y_true: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """
    Prediction Interval Coverage Probability.
    Empirical coverage — should match nominal (e.g. 0.90 for 90% PI).
    """
    return float(np.mean((y_true >= lb) & (y_true <= ub)))


def piaw(lb: np.ndarray, ub: np.ndarray) -> float:
    """
    Prediction Interval Average Width.
    Narrower = better, given coverage requirement is met.
    """
    return float(np.mean(ub - lb))


def winkler_score(
    y_true: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Winkler Score for (1−alpha) prediction interval.
    Penalises both interval width and coverage failures.
    Lower = better.

    Reference: Winkler (1972), Journal of the American Statistical Association.
    """
    width = ub - lb
    below = y_true < lb
    above = y_true > ub
    score = width.copy()
    score[below] += (2 / alpha) * (lb[below] - y_true[below])
    score[above] += (2 / alpha) * (y_true[above] - ub[above])
    return float(np.mean(score))


# ══════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════

def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    crit: str = "mse",
) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: Models 1 and 2 have equal predictive accuracy.
    DM < 0 and significant → Model 1 better than Model 2.
    DM > 0 and significant → Model 2 better than Model 1.

    Args:
        e1, e2 : forecast errors (not squared) from model 1 and model 2
        h      : forecast horizon (1 for one-step-ahead)
        crit   : loss differential ('mse' or 'mae')

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

    n    = len(d)
    dbar = float(np.mean(d))

    # Newey-West variance estimate
    gamma0 = float(np.var(d, ddof=1))
    if h > 1:
        gammas = [float(np.cov(d[k:], d[:-k])[0, 1]) for k in range(1, h)]
        V_d    = gamma0 + 2 * sum(gammas)
    else:
        V_d = gamma0

    # Harvey-Leybourne-Newbold finite-sample correction
    V_d_corrected = V_d * (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_stat = dbar / np.sqrt(max(V_d_corrected, 1e-12))

    # t-distribution with (n-1) df
    p_val = float(2 * stats.t.sf(abs(dm_stat), df=n - 1))

    return round(dm_stat, 4), round(p_val, 4)


def friedman_ranking_test(
    results_dict: dict[str, list],
    metric: str = "RMSE",
) -> tuple[float, float, dict]:
    """
    Friedman test for significant differences across multiple models.
    Non-parametric: ranks models within each fold, tests if ranks differ.

    Args:
        results_dict : {model_name: [metric_fold1, metric_fold2, ...]}
                       Values are metric scores (lower = better for RMSE).
        metric       : name label only (used in logging)

    Returns:
        (chi2_stat, p_value, mean_ranks_dict)

    Reference:
        Friedman (1937); Demsar (2006) for ML model comparison.
    """
    models = list(results_dict.keys())
    data   = np.array([results_dict[m] for m in models]).T  # (n_folds, n_models)

    # Rank within each fold (lower metric = rank 1 = best)
    ranks     = np.array([stats.rankdata(row) for row in data])
    n_folds, k = ranks.shape
    mean_ranks = ranks.mean(axis=0)

    # Friedman chi² statistic
    chi2  = (12 * n_folds / (k * (k + 1))) * (
        np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4
    )
    p_val = float(stats.chi2.sf(chi2, df=k - 1))

    mean_ranks_dict = {m: round(float(r), 3) for m, r in zip(models, mean_ranks)}
    return round(float(chi2), 4), round(p_val, 4), mean_ranks_dict


def nemenyi_cd(k: int, n_folds: int, alpha: float = 0.05) -> float:
    """
    Nemenyi post-hoc critical difference for Friedman test.
    If |mean_rank_A - mean_rank_B| > CD → significant difference.

    Critical q values (Nemenyi, α=0.05):
        k=2: 1.960, k=3: 2.343, k=4: 2.569, k=5: 2.728

    Args:
        k       : number of models being compared
        n_folds : number of cross-validation folds
        alpha   : significance level (0.05 or 0.10)

    Returns:
        critical_difference
    """
    # Studentised range distribution critical values
    q_table = {
        (2, 0.05): 1.960, (3, 0.05): 2.343, (4, 0.05): 2.569,
        (5, 0.05): 2.728, (6, 0.05): 2.850, (7, 0.05): 2.949,
        (2, 0.10): 1.645, (3, 0.10): 2.052, (4, 0.10): 2.291,
        (5, 0.10): 2.459, (6, 0.10): 2.589, (7, 0.10): 2.693,
    }
    q = q_table.get((k, alpha), 2.343)   # default: k=3, α=0.05
    return round(q * np.sqrt(k * (k + 1) / (6 * n_folds)), 4)


# ══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD SPLIT HELPER
# ══════════════════════════════════════════════════════════════════════════

def get_split_data(
    df: pd.DataFrame,
    test_year: int,
    features: list[str],
    target: str = "Y_stoch",
) -> tuple:
    """
    Return (X_train, y_train, X_test, y_test, dates_test) for one fold.

    STRICT temporal integrity:
    - train = all data BEFORE test_year
    - test  = data FROM test_year only
    - Assertion: no date overlap

    Args:
        df        : model-ready DataFrame with DATE and YEAR columns
        test_year : year to use as test set
        features  : list of feature column names
        target    : target column name

    Returns:
        X_train, y_train, X_test, y_test, dates_test
    """
    train = df[df["YEAR"] < test_year].copy()
    test  = df[df["YEAR"] == test_year].copy()

    assert len(train) > 0, f"Empty training set for test_year={test_year}"
    assert len(test) > 0,  f"Empty test set for test_year={test_year}"
    assert train["DATE"].max() < test["DATE"].min(), \
        f"Temporal leakage: train/test dates overlap for year {test_year}!"

    return (
        train[features].values,
        train[target].values,
        test[features].values,
        test[target].values,
        test["DATE"].values,
    )


# ══════════════════════════════════════════════════════════════════════════
# UNIT TEST SEDERHANA (python utils.py untuk verifikasi)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running utils self-tests...\n")

    rng = np.random.default_rng(42)
    n   = 108   # 9 folds × 12 months

    y_true = rng.normal(1.0, 0.2, n)
    y_good = y_true + rng.normal(0, 0.05, n)   # model bagus
    y_bad  = y_true + rng.normal(0, 0.20, n)   # model buruk

    # Point metrics
    print("Point metrics (good model):")
    res = evaluate_point(y_true, y_good, label="good_model")
    for k, v in res.items():
        print(f"  {k}: {v}")

    # DM test
    e_good = y_true - y_good
    e_bad  = y_true - y_bad
    dm, p  = diebold_mariano(e_good, e_bad, crit="mse")
    print(f"\nDiebold-Mariano (good vs bad): DM={dm:.3f}, p={p:.4f}")
    assert dm < 0 and p < 0.05, "Good model should beat bad model (DM test)"
    print("  ✓ DM test correct direction (good < bad)")

    # Friedman test
    fold_rmse_good = [rmse(y_true[i*12:(i+1)*12], y_good[i*12:(i+1)*12]) for i in range(9)]
    fold_rmse_bad  = [rmse(y_true[i*12:(i+1)*12], y_bad[i*12:(i+1)*12])  for i in range(9)]
    chi2, p_fr, ranks = friedman_ranking_test({
        "good": fold_rmse_good,
        "bad":  fold_rmse_bad,
    })
    print(f"\nFriedman test: chi2={chi2:.3f}, p={p_fr:.4f}")
    print(f"  Mean ranks: {ranks}")
    cd = nemenyi_cd(k=2, n_folds=9)
    print(f"  Nemenyi CD (k=2, α=0.05): {cd:.4f}")

    print("\n✅ All utils tests passed.")
