"""
=============================================================================
NOTEBOOK 01: DATA PREPROCESSING
=============================================================================
Purpose  : Load NASA POWER CSV, parse wide-format, pivot to monthly panel,
           validate data quality, export clean parquet for downstream use.
Input    : POWER_Point_Monthly_20050101_20251231_000d13N_117d50E_UTC.csv
Output   : data/01_nasa_power_clean.parquet
           data/01_data_quality_report.csv
Dependencies : pandas, nump
=============================================================================
"""

import pandas as pd
import numpy as np
import io
import os

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_CSV  = "/mnt/user-data/uploads/POWER_Point_Monthly_20050101_20251231_000d13N_117d50E_UTC.csv"
OUT_DIR  = "/home/claude/pv_research/data"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: PARSE NASA POWER WIDE FORMAT
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Parsing NASA POWER header and data block")
print("=" * 60)

with open(RAW_CSV, "r") as f:
    lines = f.readlines()

# Locate data block (after -END HEADER-)
data_start = next(i + 1 for i, l in enumerate(lines) if "-END HEADER-" in l)
df_raw = pd.read_csv(io.StringIO("".join(lines[data_start:])))

print(f"  Raw shape (wide): {df_raw.shape}")
print(f"  Parameters: {sorted(df_raw['PARAMETER'].unique())}")
print(f"  Years: {df_raw['YEAR'].min()} – {df_raw['YEAR'].max()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: PIVOT WIDE → LONG → MONTHLY PANEL
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Wide → Monthly time-series panel")

MONTHS     = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
MONTH_NUM  = {m: i + 1 for i, m in enumerate(MONTHS)}

records = []
for _, row in df_raw.iterrows():
    for m in MONTHS:
        records.append({
            "PARAMETER": row["PARAMETER"],
            "YEAR":      int(row["YEAR"]),
            "MONTH":     MONTH_NUM[m],
            "VALUE":     row[m]
        })

df_long  = pd.DataFrame(records)
df_panel = (df_long
            .pivot_table(index=["YEAR","MONTH"], columns="PARAMETER", values="VALUE")
            .reset_index())
df_panel.columns.name = None
df_panel["DATE"] = pd.to_datetime(dict(year=df_panel["YEAR"], month=df_panel["MONTH"], day=1))
df_panel = df_panel.sort_values("DATE").reset_index(drop=True)

# Replace NASA POWER missing flag with NaN
df_panel.replace(-999, np.nan, inplace=True)

print(f"  Final panel shape: {df_panel.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: RENAME COLUMNS (readable aliases)
# ══════════════════════════════════════════════════════════════════════════
RENAME = {
    "ALLSKY_SFC_SW_DWN":  "GHI",          # Global Horizontal Irradiance (kWh/m²/day)
    "ALLSKY_SFC_SW_DNI":  "DNI",          # Direct Normal Irradiance
    "ALLSKY_SFC_SW_DIFF": "DIFF",         # Diffuse Irradiance
    "ALLSKY_KT":          "KT",           # Clearness Index (dimensionless)
    "CLOUD_AMT":          "CLOUD",        # Cloud Amount (%)
    "IMERG_PRECTOT":      "PRECTOT",      # Precipitation (mm/day)
    "RH2M":               "RH",           # Relative Humidity at 2m (%)
    "T2M":                "T2M",          # Air Temperature at 2m (°C)
    "TS":                 "TS",           # Skin Temperature (°C)
    "WS10M":              "WS",           # Wind Speed at 10m (m/s)
    "WSC":                "WSC",          # Corrected Wind Speed
    "PS":                 "PS",           # Surface Pressure (kPa)
    "PSC":                "PSC",          # Corrected Surface Pressure
}
df_panel.rename(columns=RENAME, inplace=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Data quality validation")

METEO_COLS = ["GHI","DNI","DIFF","KT","CLOUD","PRECTOT","RH","T2M","TS","WS","WSC","PS","PSC"]

qc_rows = []
for col in METEO_COLS:
    s = df_panel[col]
    qc_rows.append({
        "Variable": col,
        "N":        len(s),
        "Missing":  s.isna().sum(),
        "Mean":     round(s.mean(), 4),
        "Std":      round(s.std(),  4),
        "Min":      round(s.min(),  4),
        "Max":      round(s.max(),  4),
        "Skew":     round(s.skew(), 4),
    })
df_qc = pd.DataFrame(qc_rows)
print(df_qc.to_string(index=False))

# Physical plausibility checks
assert df_panel["GHI"].between(0, 12).all(),    "GHI out of physical range"
assert df_panel["CLOUD"].between(0, 100).all(), "CLOUD > 100%"
assert df_panel["RH"].between(0, 100).all(),    "RH out of range"
assert df_panel["T2M"].between(10, 45).all(),   "T2M out of tropical range"
print("\n  ✓ All physical plausibility checks passed")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: TEMPORAL COMPLETENESS CHECK
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Temporal completeness")

expected_obs = (2025 - 2005 + 1) * 12   # 252
actual_obs   = len(df_panel)
print(f"  Expected: {expected_obs}  |  Actual: {actual_obs}")
assert actual_obs == expected_obs, f"Missing observations! Expected {expected_obs}, got {actual_obs}"
print("  ✓ Complete monthly series confirmed: 2005-01 → 2025-12")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: EXPORT
# ══════════════════════════════════════════════════════════════════════════
out_data = f"{OUT_DIR}/01_nasa_power_clean.parquet"
out_qc   = f"{OUT_DIR}/01_data_quality_report.csv"

df_panel.to_parquet(out_data, index=False)
df_qc.to_csv(out_qc, index=False)

print(f"\n  Saved: {out_data}")
print(f"  Saved: {out_qc}")
print("\n✅ Notebook 01 complete.")
print(f"   Columns available: {[c for c in df_panel.columns if c not in ['YEAR','MONTH','DATE']]}")
