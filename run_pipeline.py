"""
=============================================================================
RUN_PIPELINE.PY — Climate-Aware PV Forecasting
=============================================================================
Jalankan seluruh pipeline 10 notebook secara berurutan dalam satu perintah.

Usage:
  python run_pipeline.py              # jalankan semua tahap
  python run_pipeline.py --from 5     # mulai dari NB05
  python run_pipeline.py --only 10    # jalankan satu tahap saja
  python run_pipeline.py --dry-run    # cek dependensi tanpa eksekusi

Output:
  pipeline_run_YYYYMMDD_HHMMSS.log   # log lengkap setiap run
=============================================================================
"""

import subprocess
import sys
import os
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path

# ── Konfigurasi ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
OUT_DIR   = BASE_DIR / "outputs"
FIG_DIR   = BASE_DIR / "figures"
LOG_DIR   = BASE_DIR / "logs"

STAGES = [
    {
        "nb":   1,
        "file": "notebooks/01_data_preprocessing.py",
        "desc": "Parse NASA POWER CSV → clean monthly panel (parquet)",
        "input":  ["data/nasa_power_monthly_bontang_2005_2025.csv"],
        "output": ["data/01_nasa_power_clean.parquet",
                   "data/01_data_quality_report.csv"],
    },
    {
        "nb":   2,
        "file": "notebooks/02_target_reconstruction.py",
        "desc": "Stochastic PV target + leakage demonstration",
        "input":  ["data/01_nasa_power_clean.parquet"],
        "output": ["data/02_target_reconstructed.parquet",
                   "outputs/02_leakage_demonstration.csv",
                   "outputs/02_stochastic_target_stats.csv"],
    },
    {
        "nb":   3,
        "file": "notebooks/03_feature_engineering.py",
        "desc": "12-feature leakage-safe set + ONI/DMI + VIF",
        "input":  ["data/02_target_reconstructed.parquet"],
        "output": ["data/03_model_ready.parquet",
                   "data/03_final_features.csv",
                   "outputs/03_vif_report.csv",
                   "outputs/03_correlation_matrix.csv"],
    },
    {
        "nb":   4,
        "file": "notebooks/04_validation_framework.py",
        "desc": "Walk-forward splits + evaluation metrics",
        "input":  ["data/03_model_ready.parquet",
                   "data/03_final_features.csv"],
        "output": ["outputs/04_validation_splits.csv"],
    },
    {
        "nb":   5,
        "file": "notebooks/05_ols_hc3_model.py",
        "desc": "OLS-HC3 econometric baseline + diagnostics",
        "input":  ["data/03_model_ready.parquet",
                   "data/03_final_features.csv"],
        "output": ["outputs/05_ols_coefficients.csv",
                   "outputs/05_ols_walkforward_results.csv",
                   "data/05_ols_predictions.parquet"],
    },
    {
        "nb":   6,
        "file": "notebooks/06_sarimax_climate_model.py",
        "desc": "SARIMAX+ONI with AIC grid search",
        "input":  ["data/03_model_ready.parquet"],
        "output": ["outputs/06_sarimax_walkforward_results.csv",
                   "data/06_sarimax_predictions.parquet"],
    },
    {
        "nb":   7,
        "file": "notebooks/07_xgboost_model.py",
        "desc": "Constrained XGBoost + bootstrap PI",
        "input":  ["data/03_model_ready.parquet",
                   "data/03_final_features.csv"],
        "output": ["outputs/07_xgboost_walkforward_results.csv",
                   "data/07_xgboost_full_model.pkl",
                   "data/07_xgboost_predictions.parquet"],
    },
    {
        "nb":   8,
        "file": "notebooks/08_shap_analysis.py",
        "desc": "SHAP TreeExplainer + OLS-SHAP correspondence",
        "input":  ["data/07_xgboost_full_model.pkl",
                   "data/03_model_ready.parquet",
                   "outputs/05_ols_coefficients.csv"],
        "output": ["outputs/08_shap_values.csv",
                   "outputs/08_econometric_xai_correspondence.csv"],
    },
    {
        "nb":   9,
        "file": "notebooks/09_residual_diagnostics.py",
        "desc": "Cross-model DM/Friedman test + ENSO-residual linkage",
        "input":  ["data/05_ols_predictions.parquet",
                   "data/06_sarimax_predictions.parquet",
                   "data/07_xgboost_predictions.parquet"],
        "output": ["outputs/09_model_comparison_table.csv",
                   "outputs/09_diebold_mariano_matrix.csv",
                   "outputs/09_friedman_test_results.csv"],
    },
    {
        "nb":   10,
        "file": "notebooks/10_figure_generation.py",
        "desc": "13 publication-ready figures (150 DPI)",
        "input":  ["data/03_model_ready.parquet"],
        "output": ["figures/fig01_research_framework.png",
                   "figures/fig02_leakage_demonstration.png"],
    },
]


# ── Setup logging ────────────────────────────────────────────────────────────
def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOG_DIR / f"pipeline_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


# ── Warna terminal ───────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def ok(msg):    return f"[OK] {msg}"
def warn(msg):  return f"[WARN] {msg}"
def err(msg):   return f"[ERR] {msg}"
def info(msg):  return f"[INFO] {msg}"


# ── Cek dependensi input sebelum jalankan tahap ──────────────────────────────
def check_inputs(stage):
    missing = []
    for inp in stage.get("input", []):
        path = BASE_DIR / inp
        if not path.exists():
            missing.append(str(inp))
    return missing


# ── Jalankan satu tahap ──────────────────────────────────────────────────────
def run_stage(stage, dry_run=False):
    nb   = stage["nb"]
    file = stage["file"]
    desc = stage["desc"]
    path = BASE_DIR / file

    logging.info(f"{'='*60}")
    logging.info(f"NB{nb:02d}: {desc}")
    logging.info(f"File: {file}")

    # Cek file ada
    if not path.exists():
        logging.error(err(f"File tidak ditemukan: {file}"))
        return False

    # Cek input dependencies
    missing = check_inputs(stage)
    if missing:
        logging.error(err(f"Input hilang untuk NB{nb:02d}: {missing}"))
        logging.error(f"  Pastikan NB{nb-1:02d} sudah dijalankan lebih dulu.")
        return False

    if dry_run:
        logging.info(info(f"[DRY-RUN] Akan jalankan: {file}"))
        return True

    # Buat direktori output jika belum ada
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # Jalankan
    t_start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(BASE_DIR),
            capture_output=False,   # tampilkan output langsung di terminal
            text=True,
            timeout=1800,           # 30 menit max per tahap
        )
        elapsed = time.time() - t_start

        if result.returncode == 0:
            logging.info(ok(f"NB{nb:02d} selesai dalam {elapsed:.1f}s"))
            return True
        else:
            logging.error(err(f"NB{nb:02d} gagal (exit code {result.returncode})"))
            return False

    except subprocess.TimeoutExpired:
        logging.error(err(f"NB{nb:02d} timeout setelah 30 menit"))
        return False
    except Exception as e:
        logging.error(err(f"NB{nb:02d} error: {e}"))
        return False


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Jalankan pipeline Climate-Aware PV Forecasting"
    )
    parser.add_argument("--from",  type=int, dest="from_nb", default=1,
                        help="Mulai dari notebook ke-N (default: 1)")
    parser.add_argument("--only",  type=int, dest="only_nb", default=None,
                        help="Jalankan hanya notebook ke-N")
    parser.add_argument("--dry-run", action="store_true",
                        help="Cek dependensi tanpa eksekusi")
    args = parser.parse_args()

    log_file = setup_logging()

    print(f"\n{C.BOLD}{'='*60}")
    print(f"  Climate-Aware PV Forecasting — Pipeline Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}{C.RESET}\n")

    # Pilih tahap yang akan dijalankan
    if args.only_nb:
        stages_to_run = [s for s in STAGES if s["nb"] == args.only_nb]
        if not stages_to_run:
            print(err(f"Notebook {args.only_nb} tidak ditemukan"))
            sys.exit(1)
    else:
        stages_to_run = [s for s in STAGES if s["nb"] >= args.from_nb]

    stage_numbers = [s["nb"] for s in stages_to_run]
    print(f"  {info(f'Tahap yang akan dijalankan: {stage_numbers}')}")
    if args.dry_run:
        print(f"  {warn('DRY-RUN: tidak ada yang akan dieksekusi')}\n")

    # Jalankan
    results = {}
    total_start = time.time()

    for stage in stages_to_run:
        success = run_stage(stage, dry_run=args.dry_run)
        results[stage["nb"]] = success
        if not success and not args.dry_run:
            print(f"\n{err('Pipeline berhenti pada NB' + str(stage['nb']))}")
            print(f"  Periksa log: {log_file}\n")
            sys.exit(1)
        print()

    # Ringkasan
    total_elapsed = time.time() - total_start
    passed = sum(results.values())
    total  = len(results)

    print(f"\n{C.BOLD}{'='*60}")
    print(f"  RINGKASAN PIPELINE")
    print(f"{'='*60}{C.RESET}")
    print(f"  Berhasil : {passed}/{total} tahap")
    print(f"  Waktu    : {total_elapsed:.1f}s ({total_elapsed/60:.1f} menit)")
    print(f"  Log      : {log_file}")

    if passed == total:
        print(f"\n  {ok('Semua tahap selesai!')}")
        if not args.dry_run:
            print(f"  Output: {OUT_DIR}")
            print(f"  Figures: {FIG_DIR}")
    print()


if __name__ == "__main__":
    main()
