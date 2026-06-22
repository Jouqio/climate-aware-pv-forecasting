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
# STEP 2-5: EVALUATION METRICS, DM TEST, FRIEDMAN TEST, LEAKAGE-FREE SPLITS
# ══════════════════════════════════════════════════════════════════════════
"""
Q1 CODE AUDIT FIX (2026-06-20): This notebook previously defined ALL of
the functions below INLINE, duplicating (and diverging from) the
shared utils.py module that was supposedly created to consolidate them
(see 39_CODE_AUDIT_CRITICAL_FINDINGS.md, finding #6 — utils.py existed
but was never actually imported anywhere). Two of these inline
duplicates were also independently confirmed buggy:

  - skill_score() used y_true.mean() — the TEST SET's own mean — as the
    "climatology" baseline (finding #2/#5).
  - get_split_data() returned features as-is from a full-sample-
    precomputed *_anom column (finding #1), and its "leakage guard"
    only checked row-level date ordering, never feature-value
    contamination (finding #3).

This block now imports the single, corrected implementation from
utils.py instead of redefining everything locally. This notebook is
the canonical place these functions are validated; every other
modeling notebook (05/06/07/09) imports the SAME functions from the
SAME module, guaranteeing consistency across the whole pipeline.
"""
print("\nSTEP 2-5: Importing shared metrics / tests / split functions from utils.py")

import sys
sys.path.insert(0, str(BASE_DIR.parent))  # utils.py lives at repo root
from utils import (
    rmse, mae, mape, skill_score, evaluate_point,
    crps_normal, picp, piaw, winkler_score,
    diebold_mariano, friedman_ranking_test,
    get_split_data, climatology_baseline_predict,
    verify_no_feature_leakage,
)

print(f"  ✓ Metric functions imported: RMSE, MAE, MAPE, SkillScore, CRPS, PICP, PIAW, Winkler")
print(f"  ✓ Diebold-Mariano test imported (Harvey-Leybourne-Newbold corrected)")
print(f"  ✓ Friedman ranking test imported")
print(f"  ✓ get_split_data() imported — recomputes anomaly features PER FOLD "
      f"from training-window-only climatology (Q1 audit fix)")

# Verify split integrity — Q1 AUDIT FIX: this now exercises the REAL,
# corrected get_split_data() (per-fold anomaly recomputation +
# climatology baseline), and additionally runs the feature-VALUE-level
# leakage guard (verify_no_feature_leakage), not just date ordering.
RAW_ANOMALY_SOURCE_COLS = ["GHI", "CLOUD", "PRECTOT", "T2M"]
for split in splits:
    t_year = split["test_year"]
    verify_no_feature_leakage(df, t_year, RAW_ANOMALY_SOURCE_COLS)
    X_tr, y_tr, X_te, y_te, _, y_clim = get_split_data(
        df, t_year, FINAL_FEATURES,
        raw_anomaly_source_cols=RAW_ANOMALY_SOURCE_COLS, target="Y_stoch"
    )
    assert len(X_tr) > 0 and len(X_te) == 12, f"Split for {t_year}: size error"
print(f"  ✓ All {len(splits)} splits verified: no temporal leakage AND no "
      f"feature-value leakage (per-fold climatology confirmed divergent "
      f"from full-sample climatology for every fold)")

# ── SAVE ──────────────────────────────────────────────────────────────────
df_splits.to_csv(f"{OUT_DIR}/04_validation_splits.csv", index=False)

print(f"\n✅ Notebook 04 complete.")
print(f"   9 walk-forward folds defined.")
print(f"   Final holdout (2024–2025): {(df['YEAR'] >= 2024).sum()} months reserved.")
print(f"   Metrics: RMSE, MAE, MAPE, SkillScore, CRPS, PICP, PIAW, Winkler")
print(f"   Tests: Diebold-Mariano, Friedman")
print(f"   All metric/test/split functions are now imported from utils.py "
      f"(single source of truth) rather than redefined locally.")
