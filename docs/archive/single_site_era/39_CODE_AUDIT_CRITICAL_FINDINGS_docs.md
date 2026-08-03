# CODE AUDIT — climate-aware-pv-forecasting REPOSITORY
## Senior Scientific Software Reviewer + Q1 Associate Editor Audit
## Repository: https://github.com/Jouqio/climate-aware-pv-forecasting.git
## Audit date: 20 June 2026 (live clone, commit at audit time)

> ⚠️ **EXECUTIVE BOTTOM LINE: This repository is NOT ready for Q1
> submission. The audit uncovered a confirmed, reproducible, and
> previously undetected data leakage in feature construction, AND
> confirmed that the manuscript's central methodological contribution
> (the per-fold expanding-window climatological baseline) is NOT
> actually implemented anywhere in the committed code — not in any of
> the 10 notebooks, and not in the shared `utils.py` module either.**

This is the single most important finding from any audit performed on
this project to date. It is graded **CRITICAL — must fix before any
code-availability statement is made to a journal.**

---

# 1. EXECUTIVE SUMMARY

The repository is well-organized, documented, and shows real engineering
care in several places (hyperparameter tuning correctly confined to
fold-1 only; explicit leakage-guard assertions; physically-grounded
target construction; consolidated `utils.py` attempt). However, a
systematic, file-by-file audit — reading actual code, not assumptions —
uncovered **two independent, confirmed data-leakage defects** and **one
confirmed, repository-wide methodological inconsistency** that directly
contradicts the manuscript's central narrative:

1. **Climatological anomaly features (`GHI_anom`, `CLOUD_anom`,
   `PRECTOT_anom`, `T2M_anom`) are computed using full-sample
   (2005–2025) calendar-month means, not per-fold expanding-window
   means** — confirmed in `notebooks/03_feature_engineering.py`,
   lines 113–117. This means GHI_anom — the manuscript's single most
   important predictor (dominant in OLS and SHAP) — is contaminated
   with future information in every walk-forward fold.

2. **The walk-forward "leakage guard" only checks row-level date
   ordering, not feature-level information content** — confirmed in
   `04_validation_framework.py` and `utils.py` `get_split_data()`. The
   assertion `train["DATE"].max() < test["DATE"].min()` passes
   trivially while feature values computed in (1) are already
   contaminated. This creates **false confidence**: the code prints
   "✓ All 9 splits verified: no temporal leakage" while leakage is
   actually present.

3. **No notebook, and not `utils.py`, implements a per-fold
   expanding-window climatological baseline.** Three different,
   mutually inconsistent "SkillScore" formulas exist across
   `04/05` (test-set-std-relative), `06` (identical formula, same bug),
   `07` (identical formula, same bug), and `09` (a genuinely
   climatology-based formula — but computed on the **full 2015–2023
   test period combined**, i.e., exactly the "aggregate baseline" bug
   the manuscript explicitly says was diagnosed and corrected). The
   methodology described in Methods §3.5 and celebrated as
   Contribution C2 throughout the manuscript **does not exist in the
   code as committed.**

Additional confirmed findings: `utils.py` is fully written but **never
imported by any notebook** (dead code, and its own functions carry the
same bugs as (1)–(3) above, so importing it would not fix anything
without further correction); the target construction in
`02_target_reconstruction.py` implements **5 stochastic loss
components, not 7** (no `L_monsoon`, no `L_ENSO` term) — contradicting
the manuscript's Table 2 and Figure 6 caption; `08_shap_analysis.py`
computes SHAP **once, on the full-sample model only** — the
fold-1/fold-9 cross-window SHAP stability check repeatedly cited in the
manuscript (Figure 10B, Table S1) has no corresponding code in this
repository; and `10_figure_generation.py` still targets the **original
13-figure set including three figures already identified for removal**
(duplicate/obsolete `fig09_probabilistic_forecast`,
`fig11_ols_shap_correspondence`, `fig13_enso_phase_forecasting`).

**Net assessment:** the repository, as committed, would not reproduce
several of the manuscript's most important numbers and claims if run
end-to-end. This is a Tier-1 reproducibility crisis. The recommended
path is detailed in §6.

---

# 2. TOP 10 MASALAH KODE PALING BERBAHAYA

| # | Masalah | File | Severity |
|---|---|---|---|
| 1 | **GHI_anom/CLOUD_anom/PRECTOT_anom/T2M_anom dibangun dari klimatologi full-sample (2005–2025), bukan per-fold expanding window** — GHI_anom adalah prediktor paling dominan di seluruh manuskrip | `03_feature_engineering.py:111-117` | 🔴 CRITICAL |
| 2 | **Tidak ada implementasi baseline klimatologi per-fold di MANAPUN dalam codebase** — kontribusi inti manuskrip (C2) tidak ada kodenya | `04,05,06,07,utils.py` | 🔴 CRITICAL |
| 3 | **Leakage guard hanya memeriksa urutan tanggal baris, bukan kontaminasi nilai fitur** — assertion lolos padahal data sudah leak | `04_validation_framework.py:272-276`, `utils.py:287-288` | 🔴 CRITICAL |
| 4 | **Baseline "Climatology_baseline" di notebook 09 memakai mean test-period gabungan 2015-2023** — persis bug "aggregate baseline" yang menurut manuskrip "sudah diperbaiki" | `09_residual_diagnostics.py:124-127` | 🔴 CRITICAL |
| 5 | **Tiga "SkillScore" formula berbeda di 3 notebook model** — semuanya memakai std/mean test-set sendiri (bukan baseline klimatologi nyata), inkonsisten dengan deskripsi manuskrip | `05:235, 06:222, 07:180` | 🟠 TINGGI |
| 6 | **`utils.py` ditulis lengkap tapi TIDAK PERNAH di-import** oleh notebook manapun — dead code, dan punya bug yang sama | seluruh `utils.py` | 🟠 TINGGI |
| 7 | **Y_stoch hanya punya 5 loss component, bukan 7** seperti diklaim manuskrip (Tabel 2, Fig 6) — L_monsoon dan L_ENSO tidak ada di kode | `02_target_reconstruction.py` | 🟠 TINGGI |
| 8 | **Tidak ada fold-level SHAP stability check** (full-sample vs fold-1 vs fold-9) — klaim utama Figure 10B/Table S1 tidak punya kode pendukung | `08_shap_analysis.py` | 🟠 TINGGI |
| 9 | **Bootstrap PI XGBoost tidak menambahkan residual noise** — hanya resample data latih + refit, sehingga PI pasti undercoverage secara struktural (konsisten dgn PICP=0.361 yg dilaporkan, tapi root cause tidak didokumentasikan di kode) | `07_xgboost_model.py:160-175` | 🟡 SEDANG |
| 10 | **`notebooks/10_figure_generation.py` masih mentarget 13 figure lama**, termasuk 3 figure yang sudah diidentifikasi sebagai duplikat/obsolete di iterasi manuskrip terbaru | `10_figure_generation.py` (docstring + figure list) | 🟡 SEDANG |

---

# 3. TOP 10 PERBAIKAN PALING PENTING

| # | Perbaikan | Mengatasi # |
|---|---|---|
| 1 | **Tulis ulang fungsi anomali di NB03 agar TIDAK menghitung klimatologi sekali di awal** — pindahkan komputasi `*_anom` ke DALAM loop walk-forward, dihitung ulang per fold dari training window saja | #1 |
| 2 | **Implementasikan `expanding_climatology_baseline()` baru di `utils.py`** yang menghitung mean per calendar-month HANYA dari training fold, lalu terapkan ke test fold tersebut — gunakan ini di SEMUA notebook model (05,06,07) dan di NB09 | #2, #4, #5 |
| 3 | **Perbaiki leakage guard agar memeriksa KONTAMINASI FITUR**, bukan hanya urutan tanggal — tambahkan assertion yang membandingkan nilai fitur train-only-climatology vs full-sample-climatology dan gagal jika berbeda dari nol | #3 |
| 4 | **Hapus `utils.py` yang tidak terpakai ATAU benar-benar import dan gunakan di semua notebook** — pilih satu, jangan biarkan dead code yang membingungkan reviewer | #6 |
| 5 | **Tambahkan `L_monsoon` dan `L_ENSO` ke `02_target_reconstruction.py`** agar kode sesuai klaim manuskrip "7 loss components" — atau revisi manuskrip untuk menyebut "5 components" (instruksi mengatakan prioritaskan manuskrip → tambahkan ke kode) | #7 |
| 6 | **Tambahkan blok kode fold-1/fold-9 SHAP recomputation** di `08_shap_analysis.py` untuk benar-benar menghasilkan klaim "stability across training windows" | #8 |
| 7 | **Tambahkan residual noise ke bootstrap PI XGBoost** (`+ np.random.normal(0, residual_std, ...)` pada setiap bootstrap prediction) sebelum menghitung persentil, atau dokumentasikan eksplisit di komentar kode bahwa ini adalah epistemic-only PI dan PICP rendah diharapkan | #9 |
| 8 | **Update `10_figure_generation.py`** agar daftar figure cocok dengan 11 figure final di manuskrip (hapus fig09_probabilistic_forecast, fig11_ols_shap_correspondence, fig13_enso_phase_forecasting; tambahkan figNEW_A, figNEW_B) | #10 |
| 9 | **Tambahkan skrip/instruksi otomatis untuk download NASA POWER CSV** (saat ini raw CSV di-gitignore dan tidak ada cara otomatis mendapatkannya — pipeline TIDAK bisa dijalankan ulang dari nol tanpa langkah manual yang tidak terdokumentasi) | Reproducibility |
| 10 | **Tambahkan CI/test script kecil** yang menjalankan NB01–NB04 pada data sintetis kecil dan memverifikasi tidak ada NaN/leakage — memberi reviewer keyakinan pipeline benar-benar bisa dieksekusi ulang | Reproducibility |

---

# 4. TABEL: MASALAH → DAMPAK → PERBAIKAN

| Masalah | Dampak pada Klaim Manuskrip | Perbaikan |
|---|---|---|
| Klimatologi anomali full-sample | GHI_anom (prediktor dominan OLS p<0.001, SHAP rank-1) terkontaminasi info masa depan di SETIAP fold — klaim "GHI anomaly is dominant driver" tidak valid sebagaimana dihitung saat ini | Hitung ulang per-fold dari training window saja |
| Tidak ada baseline per-fold di kode | Klaim sentral "Contribution C2: per-fold expanding-window baseline diperlukan" — TIDAK ADA implementasi nyata untuk diverifikasi reviewer | Implementasikan fungsi baseline yang benar di `utils.py`, pakai di semua notebook |
| Leakage guard lemah | False confidence — kode mengklaim "no leakage" padahal ada | Perkuat assertion ke level fitur, bukan hanya tanggal |
| Baseline NB09 pakai test-period gabungan | Angka "Climatology_baseline" RMSE di tabel perbandingan akhir memakai persis bug yang manuskrip klaim sudah diperbaiki | Ganti dengan fungsi per-fold dari poin di atas |
| 3 formula SkillScore berbeda | Perbandingan SkillScore OLS vs SARIMAX vs XGBoost di Tabel 6 tidak apple-to-apple dengan deskripsi metodologi | Satukan ke satu fungsi baseline yang benar |
| `utils.py` tidak dipakai | Reviewer yang mengecek "apakah konsisten" akan menemukan modul yang dibuat untuk konsistensi tapi diabaikan — sinyal kurang teliti | Import & gunakan, atau hapus |
| 5 bukan 7 loss component | Tabel 2 dan Figure 6 caption di manuskrip tidak match kode aktual — discrepancy yang mudah ditemukan reviewer yang membuka repo | Tambah L_monsoon, L_ENSO ke kode |
| Tidak ada fold-level SHAP check | Figure 10B dan klaim "stability across training windows" tidak reproducible dari kode | Tambah blok kode fold-1/fold-9 SHAP |
| Bootstrap PI tanpa residual noise | PICP=0.361 benar tapi root cause (no aleatoric noise) tidak terdokumentasi — reviewer ML akan langsung melihat bug ini jika baca kode | Tambah noise term atau dokumentasikan eksplisit |
| NB10 figure list usang | Repo "official" figure generator tidak menghasilkan figure final yang ada di manuskrip — reviewer akan bingung mana yang benar | Update daftar & isi NB10 |

---

# 5. CHECKLIST FILE-BY-FILE

### `notebooks/01_data_preprocessing.py`
- **Fungsi:** Parse NASA POWER wide-format CSV → monthly panel parquet.
- **Risiko leakage:** Tidak ada — operasi murni parsing/cleaning sebelum split apapun.
- **Ketidaksesuaian manuskrip:** Tidak ada yang terdeteksi.
- **Tindakan:** ✅ Tidak ada revisi wajib. Catatan minor: raw CSV input di-gitignore tanpa skrip download otomatis — tambahkan instruksi/skrip NASA POWER API call untuk reproducibility penuh.

### `notebooks/02_target_reconstruction.py`
- **Fungsi:** Membangun Y_det (leakage proof) dan Y_stoch (target terkoreksi).
- **Risiko leakage:** Tidak ada leakage temporal (target construction terjadi sebelum split, ini benar secara desain — Y_stoch bukan hasil model, jadi tidak perlu per-fold).
- **Ketidaksesuaian manuskrip:** 🔴 **Hanya 5 loss component (L_thermal, L_cloud_resid, L_aerosol, L_humidity, L_inverter), bukan 7.** L_monsoon dan L_ENSO yang disebut di Tabel 2 manuskrip dan caption Figure 6 TIDAK ADA di kode ini.
- **Tindakan:** Tambahkan dua loss component yang hilang, ATAU revisi manuskrip menjadi "5 components" — instruksi meminta prioritaskan manuskrip, sehingga rekomendasi: **tambahkan kode**.

### `notebooks/03_feature_engineering.py`
- **Fungsi:** Membangun 12 fitur final + ONI/DMI sintetis + VIF.
- **Risiko leakage:** 🔴 **KRITIS.** Baris 111-117: klimatologi bulanan dihitung dari SELURUH dataset (2005-2025) sekali di awal, sebelum walk-forward split manapun terjadi. `GHI_anom`, `CLOUD_anom`, `PRECTOT_anom`, `T2M_anom` SEMUA terkontaminasi.
- **Ketidaksesuaian manuskrip:** Manuskrip §3.3 menyatakan "All anomaly features are computed against expanding training-window climatological means" — ini TIDAK BENAR untuk kode aktual.
- **Tindakan:** **WAJIB DIPERBAIKI SEBELUM SUBMIT.** Lihat kode pengganti di §7.

### `notebooks/04_validation_framework.py`
- **Fungsi:** Definisi split walk-forward, metrik evaluasi, DM test, Friedman test, "leakage guard."
- **Risiko leakage:** 🔴 **KRITIS** (dua bug independen). (a) `skill_score()` baris 103-113 memakai `y_true.mean()` dari TEST SET sebagai baseline — bukan klimatologi nyata. (b) `get_split_data()` baris 253-276 hanya memverifikasi urutan tanggal, TIDAK memverifikasi bahwa nilai fitur bebas dari kontaminasi — assertion "no temporal leakage" memberi false confidence.
- **Ketidaksesuaian manuskrip:** Baseline yang dideskripsikan di manuskrip §3.5 tidak ada di sini.
- **Tindakan:** **WAJIB DIPERBAIKI.** Ganti `skill_score()` dengan versi yang menghitung klimatologi per-fold dari training window.

### `notebooks/05_ols_hc3_model.py`
- **Fungsi:** OLS-HC3 full-sample + diagnostik + walk-forward.
- **Risiko leakage:** 🔴 Baris 235 — `fold_ss` memakai `y_te.mean()` (mean test set sendiri) sebagai baseline.
- **Ketidaksesuaian manuskrip:** SkillScore yang dihasilkan TIDAK sama dengan "per-fold expanding-window climatological baseline" yang dideskripsikan manuskrip.
- **Tindakan:** Ganti formula `fold_ss` dengan baseline klimatologi training-only yang benar (§7).

### `notebooks/06_sarimax_climate_model.py`
- **Fungsi:** SARIMAX + ONI, grid search order, walk-forward, analisis fase ENSO.
- **Risiko leakage:** 🟠 Baris 222 — formula `fold_ss` sama bermasalahnya dengan NB05 (std test-set sendiri). Order selection (Part A) BENAR — menggunakan hanya 2005-2014.
- **Ketidaksesuaian manuskrip:** Forecast exogenous ONI memakai nilai REALIZED (bukan forecast ONI) — ini reasonable practice untuk evaluasi retrospektif TAPI **tidak didisclosure secara eksplisit di manuskrip sebagai limitation** (kode mengakuinya di komentar, manuskrip tidak).
- **Tindakan:** Perbaiki `fold_ss`. Tambahkan kalimat limitation di manuskrip tentang "perfect foresight of ONI assumed in SARIMAX retrospective evaluation."
- **Risiko tambahan:** Baris 274 — jika SARIMAX gagal konvergen di SATU fold saja, seluruh analisis ENSO Part D di-skip TANPA peringatan eksplisit (silent failure mode).

### `notebooks/07_xgboost_model.py`
- **Fungsi:** XGBoost dengan tuning fold-1-only, walk-forward, bootstrap PI.
- **Risiko leakage:** 🟢 Hyperparameter tuning BENAR (hanya inner train/val dalam fold-1, 2005-2012 vs 2013-2014) — ini kekuatan nyata, bukan kelemahan. 🟠 Baris 180 — `fold_ss` formula sama bermasalahnya dengan NB05/06.
- **Ketidaksesuaian manuskrip:** Hyperparameter di manuskrip ("max_depth=3, lr=0.03...") adalah NILAI SPESIFIK, tapi kode melakukan grid search otomatis — nilai final bergantung hasil search aktual, bisa berbeda dari yang didokumentasikan manuskrip jika dijalankan ulang. `BOOTSTRAP_SAMPLES=50` dengan komentar "use 200 for final paper" — TIDAK PERNAH diubah ke 200.
- **Tindakan:** Perbaiki `fold_ss`. Set `BOOTSTRAP_SAMPLES=200` sebelum hasil final mana pun direport. Tambahkan residual noise ke bootstrap predictions.

### `notebooks/08_shap_analysis.py`
- **Fungsi:** SHAP TreeExplainer pada model full-sample, korespondensi OLS-XAI, analisis ENSO-stratified.
- **Risiko leakage:** Tidak ada — SHAP pada model yang sudah dilatih bukan operasi prediktif baru.
- **Ketidaksesuaian manuskrip:** 🔴 **TIDAK ADA kode untuk fold-1/fold-9 SHAP stability check** yang menjadi salah satu klaim kekuatan utama manuskrip (Figure 10B, Table S1).
- **Tindakan:** Tambahkan blok kode baru yang melatih model pada subset fold-1 (2005-2014) dan fold-9 (2005-2022), hitung SHAP masing-masing, bandingkan ranking dengan full-sample.

### `notebooks/09_residual_diagnostics.py`
- **Fungsi:** Tabel performa agregat, DM pairwise, Friedman test, analisis ENSO-residual.
- **Risiko leakage:** 🔴 **KRITIS.** Baris 124-127 — "Climatology_baseline" dihitung dari klimatologi bulanan atas SELURUH test period 2015-2023 GABUNGAN, bukan per-fold expanding window dari training data. Ini PERSIS bug "aggregate baseline" yang manuskrip eksplisit klaim sudah diperbaiki.
- **Ketidaksesuaian manuskrip:** Ini adalah inkonsistensi PALING SERIUS dalam audit ini — kode menghasilkan angka baseline menggunakan metodologi yang manuskrip sendiri menyebutnya sebagai metodologi yang SALAH.
- **Tindakan:** **WAJIB DIPERBAIKI SEBELUM SUBMIT** — lihat kode pengganti §7.

### `notebooks/10_figure_generation.py`
- **Fungsi:** Generate 13 figure untuk publikasi.
- **Risiko leakage:** Tidak relevan (visualisasi, bukan modeling).
- **Ketidaksesuaian manuskrip:** Daftar 13 figure di docstring TIDAK cocok dengan 11 figure final manuskrip — termasuk 3 figure yang sudah diidentifikasi obsolete/duplikat (fig09_probabilistic_forecast, fig11_ols_shap_correspondence, fig13_enso_phase_forecasting) dan tidak memiliki figNEW_A/figNEW_B.
- **Tindakan:** Update docstring dan kode generator agar sesuai 11-figure final set.

### `utils.py`
- **Fungsi:** Modul shared utility (dimaksudkan untuk konsolidasi fungsi dari NB04/NB09).
- **Risiko leakage:** 🔴 Memiliki BUG YANG SAMA dengan #1/#2 (skill_score baris 50, get_split_data baris 282-289 tidak recompute klimatologi).
- **Ketidaksesuaian manuskrip:** 🟠 **TIDAK PERNAH DI-IMPORT oleh notebook manapun** — dead code yang dibuat untuk konsistensi tapi tidak digunakan sama sekali.
- **Tindakan:** Perbaiki bug DAN pastikan semua notebook benar-benar mengimpor dari sini (single source of truth).

### `run_pipeline.py`
- **Fungsi:** Runner orchestration untuk 10 notebook berurutan.
- **Risiko leakage:** Tidak langsung — tapi mewarisi semua bug di atas karena hanya memanggil notebook secara berurutan.
- **Tindakan:** Tidak ada perubahan struktural diperlukan; akan otomatis benar setelah notebook individual diperbaiki.

### `fetch_oni_dmi.py`, `27_oni_integration_complete.py`, `oni_integration_complete.py`
- **Catatan:** Tiga file terkait integrasi ONI/DMI dengan nama mirip — potensi duplikasi/kebingungan struktur repo. **Rekomendasi:** konsolidasi menjadi SATU file bernama jelas (mis. `scripts/fetch_oni_dmi.py`), hapus duplikat.

### `README.md`
- **Tindakan:** Tambahkan instruksi eksplisit cara mendapatkan raw NASA POWER CSV (saat ini di-gitignore tanpa skrip download), dan tambahkan disclaimer eksplisit tentang status "Y_stoch adalah simulasi ilustratif, bukan data PV terukur" konsisten dengan framing manuskrip terbaru.

---

# 6. REKOMENDASI FINAL — APAKAH REPO SIAP UNTUK Q1?

## **TIDAK SIAP. Status: MAJOR CODE REVISION REQUIRED.**

Tiga temuan kritis (klimatologi full-sample, baseline tidak ada di kode,
leakage guard yang lemah) berarti: **jika seorang reviewer Q1 menjalankan
repository ini end-to-end dan membandingkan hasilnya dengan klaim
metodologi di manuskrip, mereka akan menemukan bahwa kode tidak
mengimplementasikan apa yang manuskrip deskripsikan sebagai kontribusi
intinya.** Ini adalah risiko reputasional dan ilmiah yang serius —
lebih serius daripada masalah framing/klaim yang sudah diperbaiki di
revisi-revisi sebelumnya, karena ini menyangkut **executable evidence**,
bukan hanya narasi teks.

**Jangan submit dengan tautan "code available at GitHub" sampai
perbaikan §3 (terutama #1-#4) diimplementasikan dan dijalankan ulang.**

Estimasi waktu perbaikan: **6-10 jam kerja fokus** (lihat kode pengganti
di §7 sebagai starting point untuk fix #1-#4).

Setelah perbaikan, jalankan ulang seluruh pipeline (`python
run_pipeline.py`), VERIFIKASI angka-angka baru terhadap manuskrip, dan
update tabel/figure yang berubah. Beberapa angka manuskrip (terutama
yang melibatkan GHI_anom sebagai prediktor dan SkillScore tabel 6)
**kemungkinan akan berubah** setelah perbaikan ini — ini diharapkan dan
benar; perubahan tersebut mencerminkan penghapusan leakage yang
sebelumnya secara tidak sengaja menginflasi performa.

---

# 7. KODE PENGGANTI — PERBAIKAN PALING KRITIS

Lihat file terpisah: **`40_utils_FIXED.py`** dan
**`41_notebook03_climatology_patch.py`** untuk kode lengkap yang siap
dipakai, dengan penjelasan inline di setiap perubahan.

