# Dokumentasi Audit & Hasil — Q1 Code Audit (2026-06)

File-file di folder ini mendokumentasikan audit kode kritis yang
dilakukan sebelum submission jurnal Q1, dan hasil menjalankan pipeline
yang telah diperbaiki pada data NASA POWER asli.

| File | Isi |
|---|---|
| `39_CODE_AUDIT_CRITICAL_FINDINGS.md` | Audit lengkap: 10 masalah kode, file-by-file checklist |
| `44_REAL_DATA_RESULTS_SUMMARY.md` | Perbandingan angka manuskrip (lama) vs hasil data asli + kode terperbaiki (baru) |
| `43_HOW_TO_APPLY_AND_PUSH.md` | Cara apply patch ke repo Anda sendiri |

## Ringkasan Commit di branch `fix/q1-leakage-audit`

```
4d11ff0 fix(leakage): eliminate confirmed data leakage + align code with manuscript
d4ea35b fix(critical): add missing low-VIF OLS-HC3 specification
93b712e fix: residual diagnostics now computed on low-VIF spec
c231b85 fix: add missing Wilcoxon test and Winkler Score computations
5d7649a fix: recompute anomaly features for descriptive plotting (notebook 10)
```

## Status

✅ Pipeline lengkap (01→10) diverifikasi berjalan end-to-end pada data
NASA POWER asli untuk Bontang (0.1333°N, 117.5°E), 2005-2025.

✅ Semua output (CSV, parquet, figure) di folder `notebooks/data/`,
`notebooks/outputs/`, `notebooks/figures/` adalah hasil run TERBARU
dengan kode yang sudah diperbaiki — bukan data sintetis testing.

⚠️ Branch ini (`fix/q1-leakage-audit`) BELUM di-merge ke `main`.
Review diff sebelum merge, terutama:
- `utils.py` (rewrite total)
- `notebooks/03_feature_engineering.py` (penghapusan leakage utama)
- `notebooks/09_residual_diagnostics.py` (perbaikan baseline kritis)
