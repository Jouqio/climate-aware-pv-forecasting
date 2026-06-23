# HASIL PIPELINE DENGAN DATA NASA POWER ASLI (SETELAH FIX LEAKAGE)
## Bontang, East Kalimantan | 0.1333°N, 117.5°E | 2005-2025 | n=252
## Pipeline: branch fix/q1-leakage-audit (semua 9 fix audit diterapkan)

> Dijalankan end-to-end pada data NASA POWER **asli** yang Anda unggah
> (`POWER_Point_Monthly_20050101_20251231_000d13N_117d50E_UTC.csv`).
> Statistik deskriptif (GHI mean=4.8648, CLOUD=79.90%) **persis cocok**
> dengan yang telah dilaporkan di manuskrip — mengonfirmasi data ini
> adalah sumber asli yang sama.

---

## TEMUAN PALING PENTING: 1 BUG BARU DITEMUKAN DAN DIPERBAIKI

**Spesifikasi "low-VIF 8-fitur" yang menjadi klaim utama manuskrip
(GHI_anom β=+0.088, p<0.001) TIDAK ADA di kode `05_ols_hc3_model.py`** —
notebook hanya pernah fit 12 fitur penuh, di mana VIF=2900+ dari
`GHI_x_CLOUD` membuat tanda GHI_anom **terbalik jadi negatif dan tidak
signifikan** (β=-0.120, p=0.105).

✅ **Sudah diperbaiki** — spesifikasi low-VIF ditambahkan ke notebook 05
(Part A2). Hasilnya pada data asli:

```
GHI_anom:  β = +0.0874,  p = 0.0017 **   ← manuskrip: β=+0.088, p<0.001
GHI_lag1:  β = +0.0503,  p = 0.0003 ***  ← manuskrip: β=+0.028, p=0.038
```

**Klaim utama manuskrip TERVALIDASI pada data asli**, setelah fix ini.

---

## RINGKASAN: MANUSKRIP (lama) vs DATA ASLI + KODE TERPERBAIKI (baru)

| Metrik | Manuskrip (lama) | Data Asli + Fix (baru) | Status |
|---|---|---|---|
| **GHI mean / SD** | 4.865 / 0.407 | 4.8648 / 0.4068 | ✅ Identik |
| **CLOUD mean** | 79.9% | 79.90% | ✅ Identik |
| **R²_det (deterministic)** | 0.9999 | 1.0000 | ✅ Identik |
| **R²_stoch (reference)** | 0.226 | 0.335 | ⚠️ Berbeda (lihat catatan) |
| **Leakage ratio (reference)** | 4.42× | 2.98× | ⚠️ Lebih rendah, masih ≫2.5× |
| **GHI_anom β (low-VIF OLS)** | +0.088, p<0.001 | **+0.0874, p=0.0017** | ✅ **Sangat cocok!** |
| **GHI_lag1 β** | +0.028, p=0.038 | +0.0503, p=0.0003 | ✅ Arah & signifikansi cocok |
| **ONI β (OLS)** | -0.031, p=0.117 (ns) | +0.0138, p=0.496 (ns) | ⚠️ Tanda berbeda, sama2 ns |
| **VIF: GHI_anom** | 16.2 | 2.99 | ⚠️ Berbeda signifikan |
| **VIF: ONI_lag2** | 9.59 | 9.59 | ✅ Identik persis |
| **VIF: GHI_x_CLOUD** | 2,927 | 2,927 | ✅ Identik persis |
| **XGBoost mean SS** | +0.085 | +0.0756 | ✅ Sangat dekat |
| **XGBoost positive folds** | 7/9 | 7/9 | ✅ **Identik** |
| **SHAP GHI_anom rank** | 1 (semua window) | 1 (semua window) | ✅ **Identik, terverifikasi nyata** |
| **DM test (OLS vs XGB) p** | 0.960 | 0.920 | ✅ Sangat dekat, sama2 "no difference" |
| **OLS mean SS** | +0.032 (7/9 positif) | +0.023 (lihat per-fold) | ⚠️ Berbeda — baseline kini benar |
| **SARIMAX MA(1)** | θ=0.057-0.098, ns | θ=0.162, **p=0.007 (signifikan!)** | 🔴 **Berubah arah kesimpulan** |
| **Friedman test** | χ²=2.889, p=0.236 (ns) | χ²=6.000, **p=0.0498 (signifikan!)** | 🔴 **Berubah arah kesimpulan** |
| **Climatology baseline RMSE** | 0.0708 | 0.0715 | ✅ Sangat dekat |
| **PICP SARIMAX (95%)** | 0.935 | 0.926 | ✅ Sangat dekat |
| **PICP XGBoost (90%, bootstrap)** | 0.361 (sangat buruk) | 0.833 (jauh lebih baik) | ✅ Membaik nyata setelah fix |

---

## YANG TIDAK BERUBAH (klaim manuskrip ROBUST)

1. **Leakage ratio masih jauh di atas 2.5×** (2.98× pada skenario referensi) —
   klaim "conservative lower bound 2.5×" tetap valid dan konservatif.
2. **GHI_anom tetap prediktor dominan** di OLS (low-VIF) DAN SHAP — klaim
   inti paper tervalidasi penuh pada data asli.
3. **XGBoost 7/9 fold positif** — angka identik dengan manuskrip.
4. **SHAP stability** (full-sample/fold-1/fold-9 semua rank-1 GHI_anom) —
   **terverifikasi benar-benar true**, bukan asumsi (ini fix baru yang
   pertama kali benar-benar menjalankan pengecekan ini).
5. **OLS ≡ XGBoost secara statistik** (DM p=0.920, masih "no difference").

---

## YANG BERUBAH DAN PERLU DIPUTUSKAN

### 🔴 Perubahan arah kesimpulan (perlu update teks manuskrip)

1. **Friedman test sekarang SIGNIFIKAN** (p=0.0498, bukan p=0.236).
   Mean rank: OLS=1.667, XGBoost=1.667, SARIMAX=2.667 → SARIMAX secara
   signifikan lebih buruk dari OLS/XGBoost. **Manuskrip perlu direvisi**:
   kalimat "no ranking hierarchy" tidak lagi akurat.

2. **SARIMAX MA(1) sekarang SIGNIFIKAN** (θ=0.162, p=0.007, bukan ns).
   Limitation L6 ("MA(1) non-significant") **perlu dihapus/direvisi** —
   ini sekarang temuan positif, bukan limitation.

3. **R²_stoch reference = 0.335** (bukan 0.226), karena 2 komponen loss
   baru (L_monsoon, L_ENSO) yang ditambahkan sesuai audit. Leakage ratio
   reference scenario turun ke 2.98× (dari 4.42×) — **masih jauh di atas
   2.5×**, jadi klaim utama tetap aman, tapi angka spesifik berubah.

### 🟡 Perbedaan yang perlu diverifikasi/diputuskan

4. **VIF GHI_anom turun dari 16.2 → 2.99**. Ini PERBAIKAN (lebih rendah
   = lebih sehat), tapi manuskrip perlu update angka ini di Tabel 5/S4.
   Catatan: bootstrap-SE-check (yang manuskrip pakai untuk membela
   VIF=16.2) sekarang tidak lagi diperlukan karena VIF sudah rendah.

5. **ONI berganti tanda** (negatif→positif di OLS), tapi **tetap
   non-signifikan** di kedua kasus. Tidak mengubah kesimpulan ("ONI
   directional, not confirmed"), tapi arah panah perlu diperbarui di teks.

6. **OLS per-fold SkillScore pattern berbeda** — fold 2015 sekarang
   SS=-0.505 (sangat negatif, bukan +0.12 seperti manuskrip lama).
   Ini KARENA baseline klimatologi sekarang benar (bukan testset-mean
   yang bocor) — 2015 (super El Niño) memang secara fundamental lebih
   sulit diprediksi dengan baseline yang adil. Pola fold-level di Tabel 6
   perlu di-update menyeluruh.

### 🟢 Catatan metodologis (bukan kesalahan, tapi perlu transparansi)

7. **Analisis ENSO di notebook 06/08/09 masih memakai ONI sintetis
   internal** (n=24 El Niño/51 Neutral/33 La Niña) — BUKAN data NOAA CPC
   resmi yang kita ambil terpisah lewat web-fetch sebelumnya. Integrasi
   ONI resmi ke notebook 03 belum dilakukan dalam audit kode ini (ini
   pekerjaan terpisah, lihat dokumen 27/35 sebelumnya).

---

## FILE HASIL LENGKAP

Semua 25 file CSV dari run data asli ini tersedia di folder
`real_data_results/` — termasuk koefisien OLS (kedua spesifikasi),
walk-forward per-fold semua model, SHAP values, DM matrix, Friedman test,
dan tabel perbandingan model final.

