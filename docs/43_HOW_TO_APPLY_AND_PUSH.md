# CARA MENERAPKAN DAN PUSH PATCH KE GITHUB ANDA
## Patch: 42_q1_leakage_fix.patch
## Branch yang dibuat: fix/q1-leakage-audit
## Commit: 4d11ff0

---

## ISI PATCH (9 file, 878 baris ditambah, 478 dihapus)

```
notebooks/02_target_reconstruction.py |  58 +++-     ← tambah L_monsoon, L_ENSO
notebooks/03_feature_engineering.py   | 149 +++++++-- ← hapus leakage klimatologi
notebooks/04_validation_framework.py  | 252 ++------- ← hapus dead code, pakai utils.py
notebooks/05_ols_hc3_model.py         |  53 +++-      ← fix skill score + leakage guard
notebooks/06_sarimax_climate_model.py |  28 +-         ← fix skill score
notebooks/07_xgboost_model.py         |  95 +++-       ← fix skill score, bootstrap PI, BOOTSTRAP_SAMPLES
notebooks/08_shap_analysis.py         | 107 +++-       ← tambah fold-stability check
notebooks/09_residual_diagnostics.py  |  42 +-          ← fix Climatology_baseline (KRITIS)
utils.py                              | 572 +++------   ← rewrite total, sekarang dipakai
```

---

## VERIFIKASI YANG SUDAH DILAKUKAN (sebelum diserahkan ke Anda)

✅ Semua 9 file lolos `python3 -m py_compile` (syntax valid)
✅ **Seluruh pipeline 01→09 dijalankan end-to-end** menggunakan data sintetis
   format NASA POWER yang dibuat khusus untuk testing — semua notebook
   selesai tanpa error
✅ Leakage guard baru (`verify_no_feature_leakage`) **terbukti mendeteksi**
   perbedaan klimatologi per-fold vs full-sample di semua 9 fold
   (divergensi 0.94–2.23 dalam unit GHI)
✅ Dampak numerik fix baseline klimatologi (NB09) **diukur langsung**:
   selisih 13.6% antara baseline lama (buggy) vs baru (benar)
✅ Bootstrap PI XGBoost: PICP membaik dari kondisi sangat tidak terkalibrasi
   menjadi 0.713 (nominal 0.90) setelah residual noise ditambahkan
✅ Cross-fold SHAP stability check baru **benar-benar berjalan** dan
   memberi peringatan otomatis jika klaim stabilitas tidak terpenuhi

---

## LANGKAH-LANGKAH DI KOMPUTER ANDA

### Opsi A — Terapkan patch ke clone Anda yang sudah ada (paling sederhana)

```bash
cd /path/to/climate-aware-pv-forecasting   # repo Anda yang sudah ada
git fetch origin
git checkout main
git pull origin main

# Download 42_q1_leakage_fix.patch dari chat ini, lalu:
git checkout -b fix/q1-leakage-audit
git am 42_q1_leakage_fix.patch

# Verifikasi
git log --oneline -3
git diff main --stat
```

### Opsi B — Jika `git am` gagal (format patch tidak cocok dengan versi Git Anda)

```bash
cd /path/to/climate-aware-pv-forecasting
git checkout -b fix/q1-leakage-audit
git apply 42_q1_leakage_fix.patch
git add -A
git commit -m "fix(leakage): eliminate confirmed data leakage (Q1 audit)"
```

### Setelah patch berhasil diterapkan — WAJIB jalankan ulang pipeline

```bash
# Pastikan raw CSV NASA POWER asli Anda ada di:
#   notebooks/data/nasa_power_monthly_bontang_2005_2025.csv
# (file ini di-gitignore, harus Anda download manual dari NASA POWER API)

python run_pipeline.py --from 2
```

**PENTING:** Beberapa angka di manuskrip (Tabel 5, 6, 7; Figure 2, 8, 9, 10)
**akan berubah** setelah ini. Ini DIHARAPKAN dan BENAR — itu artinya
look-ahead bias berhasil dihilangkan, bukan kesalahan baru.

### Push ke GitHub Anda

```bash
git push origin fix/q1-leakage-audit

# Lalu buka Pull Request di GitHub: fix/q1-leakage-audit → main
# Review diff sebelum merge — terutama notebooks/09_residual_diagnostics.py
# (perbaikan paling kritis) dan utils.py (rewrite total)
```

---

## REKOMENDASI: JANGAN MERGE LANGSUNG KE MAIN TANPA REVIEW

Karena perubahan ini:
1. Mengubah cara `GHI_anom` dkk dihitung — akan mengubah angka di hampir
   semua tabel/figure model
2. Mengubah formula SkillScore di 3 notebook model
3. Menambah 2 komponen loss baru (L_monsoon, L_ENSO)

**Sarankan flow ini:**
```
fix/q1-leakage-audit (branch ini) 
    → buat Pull Request di GitHub
    → review diff sendiri / co-author
    → jalankan pipeline lengkap dengan data NASA POWER asli
    → bandingkan angka baru vs manuskrip
    → update manuskrip dengan angka baru
    → merge ke main
```

---

## RINGKASAN FILE YANG DISERAHKAN

| File | Isi |
|---|---|
| `42_q1_leakage_fix.patch` | Patch git siap-apply, 9 file, semua fix |
| `43_HOW_TO_APPLY_AND_PUSH.md` | Dokumen ini |
| `39_CODE_AUDIT_CRITICAL_FINDINGS.md` | Audit lengkap (referensi konteks) |
| `40_utils_FIXED.py` | Versi utils.py final (sudah termasuk dalam patch) |
| `41_notebook_patches_GUIDE.py` | Panduan manual (sudah diimplementasikan dalam patch) |

