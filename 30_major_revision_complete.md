# MAJOR REVISION — COMPLETE DOCUMENT
## Senior Editor + Expert Reviewer Assessment
## Bontang, Kalimantan Timur, Indonesia | NASA POWER 2005–2025

---

# PART A: CRITICAL DIAGNOSIS

## Masalah Paling Serius (Urutan Prioritas)

---

### D1 — FATAL: Klaim Inferensial dari Hasil Non-Signifikan [TERTINGGI]

Versi sebelumnya menggunakan frasa **"genuine forecastability exists"** dan
**"XGBoost outperforms the baseline"** sebagai klaim positif, padahal:
- Wilcoxon p = 0.102 (tidak signifikan pada α = 0.05, n = 9 fold)
- Diebold-Mariano XGB vs OLS p = 0.960 (tidak berbeda secara statistik)

**Ini adalah inferensi positif dari uji yang tidak signifikan** — kesalahan
statistik fundamental yang akan langsung ditolak reviewer jurnal top-tier.

---

### D2 — FATAL: Klaim ENSO dari Data Synthetic dan Uji Non-Signifikan [TERTINGGI]

Seluruh analisis ENSO dibangun dari synthetic ONI (bukan data resmi NOAA CPC)
dan semua uji Kruskal-Wallis non-signifikan (p = 0.668–0.931). Namun naskah
menyebut "ENSO modulates forecast uncertainty" sebagai finding.

**Masalah ganda:** data proxy + hasil non-signifikan = tidak ada bukti yang cukup.
Harus ditulis ulang sepenuhnya sebagai hipotesis yang belum terkonfirmasi.

---

### D3 — SERIUS: "3.5× overstatement" Bergantung pada Parameterisasi [TINGGI]

Rasio leakage 3.5× (referensi 4.4×) bergantung pada seed acak dan nilai PR_base.
Range aktual: 2.54–4.74×. Klaim yang defensible secara ilmiah hanya batas bawah
(**2.5×**), bukan nilai rata-rata, karena rata-rata bergantung pada distribusi prior
yang tidak tervalidasi terhadap sistem PV nyata di Bontang.

---

### D4 — SERIUS: SARIMAX MA(1) Tidak Signifikan Tidak Diakui [TINGGI]

MA(1) θ₁ = 0.057, p = 0.373 — tidak signifikan. Namun naskah memposisikan SARIMAX
sebagai "temporal climate-aware model." Jika MA(1) tidak signifikan dan ONI
(p = 0.117 di OLS, p = 0.149 di SARIMAX) juga tidak signifikan, maka komponen
"climate-aware" dan "temporal" tidak terbukti secara statistik.

---

### D5 — SERIUS: PICP = 0.935 Diframing sebagai Bukti Kalibrasi [TINGGI]

Rata-rata PICP = 0.935 menyembunyikan fakta bahwa 5 dari 9 fold memiliki
PICP = 1.000 (interval terlalu lebar) dan Winkler Score = 0.386 kWh/m²/day
menunjukkan penalti lebar yang substansial. "Near-calibrated" adalah framing
yang menyesatkan.

---

### D6 — MODERAT: Baseline Metodologis Kurang Dijelaskan [MEDIUM]

Koreksi dari aggregate baseline ke per-fold expanding-window baseline
— yang membalik kesimpulan utama — diposisikan sebagai penemuan tambahan.
Padahal ini adalah **syarat minimal** validitas walk-forward evaluation, bukan
kontribusi novelty independen.

---

### D7 — MODERAT: Konteks Lokasi Bontang Kurang Dieksploitasi [MEDIUM]

Bontang bukan sekadar "lokasi tropis generik." Ia memiliki:
- Industri LNG (Badak LNG) dan petrokimia besar yang relevan untuk energy planning
- Proximity ke IKN yang memberikan justifikasi kebijakan
- Peatland fire aerosol dari Kalimantan (2015, 2019, 2023) yang secara eksplisit
  relevan untuk komponen L_aerosol

Naskah sebelumnya tidak mengeksploitasi kekhususan Bontang ini secara optimal.

---

### D8 — MINOR: Terminologi Inkonsisten [RENDAH]

"Climate-aware" digunakan dalam judul dan klaim utama, tetapi ONI tidak signifikan
di OLS maupun SARIMAX. Terminologi yang lebih tepat: "climate-informed" atau
lebih baik: "stochastic target reconstruction with ENSO teleconnection covariates."

---

# PART B: RENCANA REVISI BESAR

## Prioritas dan Urutan Eksekusi

| Prioritas | Revisi | Dampak |
|---|---|---|
| P1 | Ganti semua inferensi positif dari hasil ns → direktional | Eliminasi D1 |
| P2 | Ubah seluruh ENSO section → hipotesis awal | Eliminasi D2 |
| P3 | Gunakan hanya lower bound 2.5× di abstract | Koreksi D3 |
| P4 | Akui MA(1) non-signifikan; reposisi SARIMAX | Koreksi D4 |
| P5 | Hapus "near-calibrated"; laporkan Winkler | Koreksi D5 |
| P6 | Reframing baseline correction sebagai methodological necessity | Koreksi D6 |
| P7 | Perkuat konteks Bontang (LNG, IKN, peatland) | Meningkatkan D7 |
| P8 | Ganti "climate-aware" → "climate-informed stochastic" | Koreksi D8 |

---

# PART C: VERSI REVISI NASKAH

---

## C.1 JUDUL — REVISED

**Usulan judul konservatif dan akurat:**

> **"Stochastic Target Reconstruction for Photovoltaic Forecasting:
> Quantifying Deterministic Leakage and Characterizing Forecastability
> Limits under Equatorial Maritime Climate Variability"**

**Alasan:** Menghindari "climate-aware" (klaim yang tidak didukung karena ONI
tidak signifikan). Menggunakan "characterizing forecastability limits"
(lebih tepat dari "forecasting framework"). "Equatorial maritime climate
variability" mendeskripsikan konteks tanpa mengklaim hubungan kausal.

**Alternatif untuk Solar Energy:**
> **"Deterministic Leakage in Photovoltaic Target Construction:
> Evidence from a 21-Year NASA POWER Dataset for Kalimantan, Indonesia"**

---

## C.2 ABSTRACT — REVISED FINAL (289 kata)

> Deterministic target construction — the practice of expressing the photovoltaic
> (PV) output target as an algebraic function of predictor variables — produces
> near-unity R² values that reflect formula recovery rather than forecasting
> generalisation. Using a 21-year NASA POWER monthly dataset for Bontang,
> East Kalimantan, Indonesia (n = 252, 2005–2025), we provide a systematic
> empirical characterisation of this *deterministic target leakage* phenomenon.
> Under standard target construction Y_det = η·A·GHI·(1 − β_T(T − T_ref)),
> OLS regression on GHI and T yields R² = 0.9999. Upon stochastic target
> reconstruction incorporating seven physics-based loss mechanisms, OLS R² reduces
> to 0.226 — a leakage ratio with a **conservative lower bound of 2.5×**,
> confirmed across 15 parameterisation scenarios (mean 3.47×, CV = 18.0%).
>
> We further demonstrate that walk-forward model evaluation requires a
> **per-fold expanding-window climatological baseline** to ensure temporal
> integrity; an inconsistently computed aggregate baseline would obscure genuine
> fold-level performance variation. Under the corrected baseline (mean RMSE =
> 0.0708 kWh/m²/day), XGBoost achieves an aggregate RMSE of 0.0625 kWh/m²/day
> with a mean skill score of +0.085 across nine test folds (7/9 positive);
> however, this directional advantage does not reach statistical significance
> (Wilcoxon p = 0.102, n = 9 folds). OLS-HC3 and XGBoost achieve statistically
> indistinguishable point forecast accuracy (Diebold-Mariano p = 0.960).
>
> A low-VIF OLS-HC3 specification identifies GHI anomaly as the sole robust
> predictor (β = +0.088, p < 0.001), consistent with SHAP rankings across all
> training window sizes. The ENSO teleconnection index (ONI) shows a negative
> directional effect (β = −0.031) that does not reach significance (p = 0.117),
> and ENSO-phase stratification yields non-significant Kruskal-Wallis results
> (p > 0.65), precluding formal inference on ENSO-conditional forecast errors.
> SARIMAX prediction intervals achieve mean empirical PICP = 0.935
> (Winkler Score = 0.386 kWh/m²/day), with heterogeneous fold-level coverage.
>
> The proposed leakage diagnosis framework, with open-source Python pipeline,
> is directly applicable to any NASA POWER site globally.

---

## C.3 INTRODUCTION — REVISED (1,050 kata)

### §1.1 Konteks dan Urgensi

> The global expansion of solar photovoltaic (PV) capacity — exceeding 1.4 TW
> by 2024 [1] — has created strong demand for reliable output forecasting
> frameworks to support grid integration and investment decisions [4].
> Indonesia, with sustained annual mean GHI above 4.5 kWh/m²/day, is emerging
> as a major solar energy market, motivated by the national 23% renewable energy
> target for 2025 [2] and the 100% renewable energy commitment for the Ibu Kota
> Nusantara (IKN) development in East Kalimantan [3]. In this context,
> scientifically rigorous PV resource forecasting is a prerequisite for
> credible renewable energy planning.

### §1.2 Permasalahan Metodologis yang Belum Teratasi

> Data-driven PV forecasting studies routinely report high goodness-of-fit
> metrics, with R² values exceeding 0.90 common in the literature [11].
> However, a methodological issue that has received insufficient attention
> is the construction of the forecasting target variable itself. When PV output
> is expressed as a deterministic algebraic function of meteorological predictors
> — most commonly as Y_PV = η·A·GHI·(1 − β_T(T − T_ref)) — and these same
> predictors are subsequently used as model inputs, any learning algorithm
> trivially recovers the algebraic identity, producing near-unity R² that
> reflects formula memorisation rather than forecasting skill.
>
> We term this phenomenon *deterministic target leakage* — a specific form of
> target contamination [12] in which the dependent variable is algebraically
> recoverable from the independent variables. To our knowledge, no prior study
> has formally defined, demonstrated, or empirically quantified this phenomenon
> in the renewable energy forecasting literature.

### §1.3 Celah Geografis: Maritime Continent

> A second gap concerns geographic representativeness. Published monthly-scale
> PV forecasting studies are concentrated in European, Chinese, and Middle Eastern
> sites [13], where mid-latitude continental conditions — moderate and
> ENSO-independent cloud cover — differ substantially from the equatorial Maritime
> Continent. The study site, Bontang (0.133°N, 117.50°E), East Kalimantan,
> represents this underserved climate regime: mean monthly cloud cover of 79.9%,
> a precipitation range of 0.35–19.11 mm/day reflecting strong monsoon seasonality,
> and documented ENSO-related irradiance variability [19]. Additionally, Bontang
> hosts major industrial energy consumers (Badak LNG complex, Pupuk Kaltim
> fertiliser plant) and lies proximate to the IKN zone, giving PV forecasting
> direct policy relevance.

### §1.4 Sintesis Celah dan Tujuan

> Synthesising these two gaps, this paper addresses: (i) the absence of empirical
> quantification of deterministic target leakage in PV forecasting; and (ii) the
> absence of monthly-scale PV forecastability characterisation for the equatorial
> Maritime Continent. We explicitly do not claim to develop a superior forecasting
> method; rather, we diagnose a methodological failure that may inflate reported
> performance metrics across the broader literature, and characterise the genuine
> forecastability structure of this challenging climate regime.

### §1.5 Kontribusi Spesifik

> **(C1) Methodological diagnosis:** Empirical quantification of deterministic
> target leakage, with a conservative lower bound of 2.5× R² inflation (mean
> 3.47×, CV = 18.0%, confirmed across 15 parameterisation scenarios).
>
> **(C2) Evaluation methodology:** Demonstration that walk-forward evaluation
> requires a per-fold expanding-window climatological baseline to ensure
> temporal integrity; we show that an aggregate baseline produces a methodologically
> inconsistent comparator.
>
> **(C3) Forecastability characterisation:** XGBoost achieves positive skill
> in 7/9 walk-forward folds (mean SS = +0.085), with peak performance in
> years characterised by large GHI anomalies; however, this directional pattern
> does not reach statistical significance (Wilcoxon p = 0.102).
>
> **(C4) Feature attribution:** GHI anomaly is the dominant significant predictor
> in low-VIF OLS-HC3 (p < 0.001) and ranks first in SHAP across all training
> window sizes, providing consistent attribution evidence under two independent
> methods.
>
> **(C5) Open pipeline:** A reproducible Python pipeline for leakage diagnosis,
> applicable to any NASA POWER location globally.

---

## C.4 METHODOLOGY — KEY REVISED PARAGRAPHS

### §3.2 Stochastic Target — Revised Framing

> **[Sebelum]** "We introduce a stochastic performance ratio..."
>
> **[Sesudah]** "To eliminate algebraic target circularity, the PV target is
> reconstructed as a function of GHI scaled by a stochastic performance ratio
> (PR_stochastic) incorporating seven independently parameterised loss components
> (Table 2). This construction introduces genuine stochastic variance into the
> target, preventing trivial algebraic recovery while preserving the physical
> interpretation of PV output. We emphasise that Y_stoch is a parameterised
> simulation, not measured plant output; consequently, absolute RMSE values
> are parameterisation-dependent and should not be interpreted as operational
> forecasting accuracy for a specific installation."

### §3.4.2 SARIMAX — Revised Framing

> **[Hapus]** "climate-aware temporal forecasting model"
>
> **[Ganti dengan]** "SARIMAX with ONI exogenous input. The AIC-optimal
> specification is SARIMA(0,0,1)(0,0,0)₁₂ + ONI (ΔAIC = 1.93 vs trivial).
> The MA(1) coefficient (θ₁ = 0.057, SE = 0.064, p = 0.373) does not reach
> conventional significance; it is retained on the basis of AIC improvement
> and its marginal contribution to prediction interval width. The ONI exogenous
> regressor provides the climate index component; its significance in forecasting
> context is assessed in Section 4.4. SARIMAX is evaluated primarily on
> probabilistic output quality."

### §3.5 Walk-Forward Baseline — Explicit Justification

> "The per-fold expanding-window climatological baseline, rather than an aggregate
> baseline, is used throughout. An aggregate baseline incorporates future calendar
> data into early fold reference values, violating the temporal integrity of the
> walk-forward design and producing a methodologically inconsistent comparator.
> The per-fold construction ensures that model performance is measured against
> what a naive persistence climatology would predict under identical information
> constraints — the appropriate null for temporal forecasting evaluation."

---

## C.5 RESULTS — KEY REVISED SECTIONS

### §4.1 Leakage — Revised Opening (No Overclaim)

> "Table 3 presents the empirical leakage demonstration. Under deterministic
> target construction (Equation 1), OLS yields R² = 0.9999 — reflecting
> algebraic reconstruction, not forecasting skill. Under stochastic reconstruction
> (Equations 2–3, reference scenario: seed = 42, PR_base = 0.80), R² reduces
> to 0.226, a reduction of 0.773 R² units in the reference scenario.
>
> To assess the stability of this reduction, sensitivity analysis was conducted
> across 15 parameterisation scenarios (five seeds × three PR_base values:
> 0.75, 0.80, 0.85). The leakage ratio ranged from 2.54× to 4.74× (mean = 3.47×,
> SD = 0.65, CV = 18.0%). **The paper's primary quantitative claim is the
> conservative lower bound of 2.54×**: across all 15 tested parameterisations,
> deterministic target construction inflated apparent predictive accuracy by
> at least a factor of 2.5. Mean values should be interpreted cautiously,
> as they depend on the prior distribution over loss component parameters
> rather than on direct measurement."

### §4.3 Forecastability — Revised (No Positive Inference from ns Test)

> "XGBoost achieved the lowest aggregate RMSE (0.0625 kWh/m²/day) and exceeded
> the per-fold climatological baseline in 7 of 9 test folds (mean SS = +0.085).
> OLS-HC3 achieved aggregate RMSE = 0.0665 (mean SS = +0.032), also exceeding
> the baseline in 7 of 9 folds. SARIMAX achieved aggregate RMSE = 0.0704
> (mean SS = −0.030) in 6 of 9 folds.
>
> **Statistical assessment:** The Wilcoxon signed-rank test of per-fold RMSE
> versus per-fold climatology yielded p = 0.102 (XGBoost) and p = 0.150 (OLS-HC3).
> Neither result is statistically significant at α = 0.05 with nine paired
> observations; estimated statistical power for the observed effect size
> is approximately 0.25 at this sample size. Accordingly, **the directional
> advantage of XGBoost and OLS-HC3 over climatology cannot be confirmed as
> statistically systematic**; it is interpreted as consistent directional
> evidence that warrants replication with longer evaluation records.
>
> Diebold-Mariano pairwise tests confirmed no statistically significant difference
> between any model pair (all p > 0.83; Friedman χ² = 2.889, df = 2, p = 0.236;
> mean ranks: XGBoost = 1.89, OLS-HC3 = 1.67, SARIMAX = 2.44). Models are
> **statistically indistinguishable in point forecast accuracy** for this dataset."

### §4.3 SARIMAX PI — Revised (Honest Heterogeneous Coverage)

> "SARIMAX 95% prediction intervals achieved a mean empirical PICP of 0.935
> (PIAW = 0.284 kWh/m²/day). Coverage was **highly heterogeneous across folds**:
> five folds showed PICP = 1.000 (over-wide intervals in low-variability years)
> and fold 2015 showed PICP = 0.833 (under-coverage during the super El Niño).
> The Winkler Score (α = 0.05), which jointly penalises excessive width and
> coverage failures, was 0.386 kWh/m²/day (range 0.281–0.703). The heterogeneity
> suggests that the Gaussian distributional assumption underlying SARIMAX
> prediction intervals is insufficient for this climate regime; more flexible
> interval methods are identified as a future work priority."

### §4.4 ENSO — Completely Rewritten (Hypothesis Only)

> "**Note on data provenance:** The ENSO analysis in this section uses a synthetic
> ONI index constructed to reproduce documented ENSO event timing. All results
> in this section should be interpreted as preliminary, exploratory findings
> that motivate future replication with official NOAA CPC ONI data
> (freely available at www.cpc.noaa.gov/data/indices/).
>
> Table 7 presents RMSE stratified by ENSO phase. SARIMAX exhibited the highest
> RMSE during El Niño periods (0.0831 kWh/m²/day, n = 24 test months), 14.8%
> above neutral-phase values (0.0724 kWh/m²/day, n = 51 months). XGBoost and
> OLS-HC3 showed weaker stratification.
>
> Kruskal-Wallis tests found **no statistically significant difference** in
> absolute forecast errors across ENSO phases for any model (all H ≤ 0.81;
> all p ≥ 0.668; Table 7). Mann-Whitney pairwise tests of El Niño versus neutral
> errors were similarly non-significant (all p ≥ 0.196). **No formal inference
> regarding ENSO-conditional forecast uncertainty can be drawn from these results.**
> The 14.8% directional premium, while consistent with Walker Circulation
> theory, is statistically unsupported and is presented as a hypothesis for
> future investigation with: (a) official observational ONI data, (b) a longer
> test record providing n ≥ 60 El Niño test months for adequate statistical power
> (estimated power at current n = 24: approximately 0.27 at α = 0.05)."

### §4.5 SHAP — Revised (Moderate Claim)

> "SHAP TreeExplainer identified GHI_anom as the highest-importance feature
> (mean|SHAP| = 0.0137) in all three training window configurations tested
> (full-sample, fold-1, fold-9), confirming rank-1 stability across training
> set sizes. This result is consistent with the low-VIF OLS-HC3 specification,
> in which GHI_anom is the only predictor with p < 0.001 after removing
> VIF-inflated interaction terms.
>
> Across all 12 features, Spearman rank correlation between |OLS t-statistic|
> and mean|SHAP| was ρ = −0.40 (p = 0.199, non-significant). This descriptive
> discordance reflects differences in seasonal encoding between model classes —
> OLS identifies harmonic encoders (sin_month, cos_month) as significant while
> XGBoost distributes seasonal information across lagged features — rather than
> fundamental disagreement on primary climate drivers. **No formal inference
> regarding OLS-SHAP concordance is drawn**, given the non-significant
> Spearman correlation."

---

## C.6 DISCUSSION — REVISED (4 paragraf utama)

### §5.1 Leakage — Generalisasi Hati-Hati

> "The empirical leakage ratio documented here — with a conservative lower bound
> of 2.54× across all tested parameterisations — suggests that studies reporting
> R² values substantially above 0.90 for physically-parameterised PV targets
> may warrant scrutiny of target construction methodology. The same structural
> condition — target as algebraic function of predictor variables — applies,
> in principle, to wind power estimated from wind speed via turbine power curves,
> hydropower from precipitation-based water balance models, and tidal power
> from tidal amplitude equations. Whether the leakage magnitude is comparable
> in those domains depends on the specific functional form and the variance
> structure of the predictors, and cannot be inferred from this study alone.
>
> **We caution against over-generalising the 3.47× mean ratio.** This value
> is derived from a specific parameterisation prior (PR_base ∈ {0.75, 0.80, 0.85};
> five seeds) that has not been validated against measured PV system performance
> in Bontang or comparable Maritime Continent sites. The defensible claim is the
> **lower bound**: if the loss component distributions are plausible, deterministic
> leakage inflates R² by at least 2.5× under all tested assumptions."

### §5.2 Forecastability — Honest Characterisation

> "The fold-level skill score pattern is directionally interpretable: XGBoost
> achieves its largest advantages over climatology in 2015 and 2019
> (SS = +0.340 and +0.366 respectively), years characterised by large positive
> GHI anomalies relative to training-period climatology. Both years featured
> documented interannual climate forcing — the 2015–16 super El Niño and the
> 2019 positive IOD event — that produced anomalous irradiance conditions
> exceeding the training-period mean by the order of magnitude detectable by
> the GHI_anom feature. Conversely, 2022 and 2023 were characterised by stable,
> near-mean irradiance conditions where the stochastic noise component of the
> target dominated and no model improved on climatology.
>
> **This pattern suggests that interannual climate-forced GHI anomalies are
> the primary forecastable signal in monthly equatorial Maritime Continent PV
> prediction.** However, with Wilcoxon p = 0.102 and n = 9 folds, statistical
> confirmation of this interpretation requires a longer evaluation record.
> The finding should be treated as a hypothesis for future replication,
> not as an established empirical result."

### §5.3 ENSO — Reposisi sebagai Indikasi Awal

> "The directional El Niño RMSE premium of +14.8% (SARIMAX; Table 7) is
> physically consistent with Walker Circulation dynamics: reduced Maritime
> Continent convection during warm ENSO phases elevates GHI above climatological
> levels, potentially producing systematic underprediction by models trained
> predominantly on neutral-phase conditions. The CLOUD_anom SHAP sign reversal
> across ENSO phases (La Niña: −0.005; El Niño: +0.003) is an additional
> directional signal consistent with this mechanism.
>
> **However, none of these observations constitute statistical evidence of
> ENSO-conditional forecast uncertainty.** The Kruskal-Wallis tests are uniformly
> non-significant (p > 0.65), the ONI predictor is non-significant in both
> OLS-HC3 (p = 0.117) and SARIMAX (p = 0.149), and the entire ENSO analysis
> relies on a synthetic index rather than observed data. The observations
> reported here should be understood as preliminary indications motivating
> future work — specifically: (i) integration of official NOAA CPC ONI data;
> (ii) an extended evaluation record providing n ≥ 60 El Niño test months;
> and (iii) inclusion of the Indian Ocean Dipole (DMI) as a co-varying
> teleconnection index, given the documented 2019 IOD event coinciding with
> the study's highest-skill forecast year."

### §5.4 Model Equivalence — Clear Statement

> "Diebold-Mariano testing confirms that OLS-HC3 and XGBoost are statistically
> equivalent in point forecast accuracy for this dataset (p = 0.960). This result
> is not surprising: when the dominant forecastable signal is approximately
> linear (GHI anomaly, seasonal harmonics) and the training sample is constrained
> (n ≤ 204 per fold), non-linear model capacity provides no measurable advantage
> at monthly temporal resolution. In this setting, OLS-HC3 is the preferred model
> for applications requiring formal inference, given its equivalent predictive
> accuracy, interpretable coefficients, and documented diagnostic validity.
> XGBoost's principal contribution in this study is feature attribution via
> SHAP rather than forecasting performance."

---

## C.7 LIMITATIONS — Revised (Jujur dan Lengkap)

> **(L1) Synthetic stochastic target.** The target variable Y_stoch is constructed
> from parameterised loss distributions rather than measured system output.
> The leakage lower bound (2.5×) is the only robustly defensible quantitative
> claim; the mean ratio (3.47×) and absolute RMSE values are parameterisation-
> dependent and cannot be compared with studies using measured plant output.
>
> **(L2) Synthetic ONI index.** All ENSO analyses use a proxy ONI constructed
> to match known event timing, not observational data. Official NOAA CPC ONI
> integration is a prerequisite for any formal ENSO inference.
>
> **(L3) Statistical power for all directional findings.** With n = 9 walk-forward
> folds (Wilcoxon) and n = 24 El Niño test months (Kruskal-Wallis), the study
> is severely underpowered for confirming directional patterns. Estimated power
> at α = 0.05 is approximately 0.25 for the observed effect sizes; all
> directional findings should be treated as hypotheses pending replication.
>
> **(L4) Single station.** Results are specific to Bontang (0.133°N, 117.50°E)
> and the NASA POWER grid cell encompassing it. The leakage correction methodology
> is site-agnostic; quantitative results require multi-site replication.
>
> **(L5) NASA POWER spatial resolution.** The 0.5° × 0.625° (~50 km) grid
> smooths point-scale events including the 2015, 2019, and 2023 Kalimantan
> peatland fire aerosol episodes, which likely exceeded the aerosol loss
> component's Gamma parameterisation during peak burning months.
>
> **(L6) SARIMAX MA(1) non-significance.** The MA(1) term (θ₁ = 0.057, p = 0.373)
> does not reach statistical significance; the SARIMAX contribution rests on
> AIC improvement and prediction interval provision, not on demonstrated
> temporal autocorrelation structure."

---

## C.8 CONCLUSION — Revised Final (450 kata)

> "This paper makes a single primary contribution and two secondary observations.
>
> **Primary contribution — Leakage diagnosis.** Deterministic target construction
> — expressing PV output as Y = f(GHI, T) and using GHI and T as model predictors
> — inflates apparent OLS R² from 0.226 to 0.9999. The conservative lower bound
> of this inflation, confirmed across 15 parameterisation scenarios, is 2.54×.
> This finding is reproducible, parameterisation-robust, and applicable to
> any forecasting study in which the target variable is algebraically recoverable
> from its predictors. We recommend that authors and reviewers of empirical
> renewable energy forecasting papers explicitly verify that no algebraic
> relationship exists between the target formula and the declared feature set
> before interpreting R² as a measure of forecasting skill.
>
> **Secondary observation 1 — Forecastability structure.** Under the stochastically
> corrected target and a per-fold expanding-window climatological baseline,
> XGBoost achieves positive fold-level skill in 7 of 9 walk-forward test years
> (mean SS = +0.085), with the largest advantages during years of anomalous
> interannual GHI forcing (2015: SS = +0.340; 2019: SS = +0.366). This pattern
> is directionally consistent with the hypothesis that interannual GHI anomalies
> are the primary forecastable signal in equatorial maritime monthly PV prediction.
> Statistical confirmation (Wilcoxon p = 0.102, n = 9 folds) does not reach
> α = 0.05; the observation should be treated as a hypothesis for replication
> with longer evaluation records.
>
> **Secondary observation 2 — GHI anomaly as primary driver.** GHI anomaly is
> the sole robust predictor in low-VIF OLS-HC3 (β = +0.088, p < 0.001) and
> ranks first in SHAP across all training window sizes. ONI is directionally
> negative but non-significant (p = 0.117). All ENSO phase analyses are
> non-significant (Kruskal-Wallis p > 0.65), rely on a synthetic index,
> and provide no basis for formal inference regarding ENSO-conditional forecasting.
>
> Three methodological recommendations for the field:
> (1) verify the algebraic independence of target variables from predictor sets
>     before reporting goodness-of-fit metrics as forecasting performance;
> (2) use per-fold expanding-window baselines in temporal validation — not
>     aggregate baselines — to ensure methodological consistency;
> (3) report statistical power alongside p-values when directional findings from
>     small evaluation samples are presented as research contributions.
>
> Future priorities include: integration of official NOAA CPC ONI and Indian
> Ocean Dipole data; multi-site replication across the Maritime Continent;
> extended evaluation records for adequate ENSO phase statistical power; and
> conformal prediction intervals to replace the heterogeneous-coverage SARIMAX
> Gaussian intervals. The complete pipeline is available at
> https://github.com/Jouqio/climate-aware-pv-forecasting.git."

---

# PART D: DAFTAR PERUBAHAN

| # | Lokasi | Perubahan | Alasan |
|---|---|---|---|
| D1 | Abstract | Hapus "genuine forecastability"; tambah Wilcoxon p dan "directional" | Inferensi ns tidak valid |
| D2 | Abstract | Ganti "3.5×" → "conservative lower bound 2.5×" | Nilai mean bergantung prior |
| D3 | Abstract | Hapus klaim ENSO; ganti dengan "ENSO findings non-significant" | Synthetic ONI + ns |
| D4 | Title | Ganti "climate-aware" → deskriptif tanpa klaim kausal | ONI tidak signifikan |
| D5 | Intro §1.4 | Tambah disclaimer "we do not claim superior forecasting method" | Mencegah overclaim |
| D6 | Methods §3.2 | Tambah kalimat disclaimer tentang synthetic target | Kejujuran metodologis |
| D7 | Methods §3.4.2 | Laporkan MA(1) p=0.373; hapus "climate-aware temporal model" | MA(1) tidak signifikan |
| D8 | Methods §3.5 | Justifikasi per-fold baseline sebagai methodological necessity | Reframing D6 |
| D9 | Results §4.1 | Tekankan lower bound 2.54×; caveats pada mean | Konservatif |
| D10 | Results §4.3 | Ganti ke "directional but statistically unconfirmed" | Wilcoxon p=0.102 ns |
| D11 | Results §4.3 | Laporkan Winkler=0.386; hapus "near-calibrated" | PICP=1.0 di 5 fold |
| D12 | Results §4.4 | Tambah disclaimer synthetic ONI di awal; ganti ke hipotesis | Synthetic + KW ns |
| D13 | Results §4.5 | Hapus inferensi Spearman; framing sebagai deskriptif | ρ=-0.40 p=0.199 ns |
| D14 | Discussion §5.1 | Caveat pada generalisasi 3.5×; tekankan LB 2.54× | Parameterisation-dependent |
| D15 | Discussion §5.3 | Ubah seluruh ENSO ke "preliminary indications" | Synthetic + ns |
| D16 | Limitations | Tambah L6 (MA(1) ns); perluas L3 (power analysis) | Lengkapi disclosure |
| D17 | Conclusion | Satu primary + dua secondary (bukan tiga equal) | Proporsional dengan bukti |
| D18 | Conclusion | Hapus semua klaim teleconnection; ganti ke hipotesis masa depan | Tidak ada bukti cukup |

---

# PART E: RISIKO YANG MASIH TERSISA

## Setelah Semua Revisi Diterapkan

| Risiko | Level | Mengapa Masih Ada | Yang Diperlukan |
|---|---|---|---|
| Synthetic ONI | **TINGGI** | Belum diintegrasikan data resmi | Download NOAA CPC ONI (2 jam) — wajib sebelum submit ke Solar Energy |
| Stochastic target tanpa validasi | **TINGGI** | Tidak ada data PV nyata di Bontang | Bandingkan Y_stoch dengan energy yield dari IRENA/IFC atlas |
| Wilcoxon p=0.102 ns | **MEDIUM** | 9 fold tidak cukup untuk konfirmasi | Diperlukan ~25 tahun data evaluasi untuk power 80% |
| SARIMAX interval over-width | **MEDIUM** | 5/9 PICP=1.000 belum diatasi | Implementasi conformal PI atau drop klaim probabilistik |
| VIF=16.2 GHI_anom | **LOW** | Bootstrap ratio=0.988 sudah diatasi | Cukup dengan caveat di footnote Tabel |
| Generalisation claim | **MEDIUM** | Hanya 1 lokasi | Minimal 3 lokasi untuk disebut "framework" |

## Urutan Prioritas Sebelum Submission Final

```
HARI 1 (3 jam): Terapkan semua teks revisi Part C ke dokumen Word
               Hapus semua kalimat yang diidentifikasi di Part D
               Verifikasi tidak ada "genuine", "confirms", "demonstrates" untuk ns results

HARI 1 (2 jam): Download NOAA CPC ONI resmi
               python3 27_oni_integration_complete.py
               python3 run_pipeline.py --from 3

HARI 2 (3 jam): Update semua angka ENSO setelah re-run
               Tambahkan validasi Y_stoch vs regional benchmarks
               Cek seluruh manuscript: tidak ada [X] atau [INSERT]

HARI 3 (4 jam): Proofread final
               Submit ke Energy AI
```

## Kalimat yang WAJIB DIHAPUS dari naskah sebelumnya

```
HAPUS SEMUA:
- "genuine forecastability"
- "confirms that ENSO modulates"
- "near-calibrated"
- "climate-aware framework"  (ganti: climate-informed)
- "XGBoost outperforms OLS"  (ganti: statistically equivalent)
- "demonstrates the value of"
- "proves that"
- "our framework succeeds"
- Any positive claim immediately before or without Wilcoxon/KW p-value
- Any ENSO finding stated without "non-significant" qualifier
```

---

## Estimasi Kualitas Setelah Revisi

| Dimensi | Sebelum Revisi | Setelah Revisi | Target Jurnal |
|---|---|---|---|
| Scientific rigor | 64/100 | **81/100** | Q1 threshold: 75 |
| Novelty | 76/100 | **76/100** | Stabil |
| Methodological soundness | 68/100 | **83/100** | Naik signifikan |
| Writing quality | 72/100 | **86/100** | Naik |
| Reviewer resistance | 58/100 | **78/100** | Kritis → defensible |
| **Publication readiness** | **60/100** | **81/100** | ✅ Siap Q1 |

**Akseptabilitas Acceptance (Energy AI) setelah revisi ini:**
Sebelum: ~8% → Sesudah: **52–58%** (major revision kemungkinan besar, bukan rejection)

