"""
SAINT — Data Preparation Pipeline
Well: 15/9-F-12 (Volve field, Norwegian Continental Shelf)

Run:
    python prepare_f12_data.py

Output:
    f12_clean.csv          — full cleaned dataset with window labels
    f12_clean_stats.txt    — summary report of every cleaning decision

CHANGE FROM v1:
    AVG_DP_TUBING removed from ENV_FEATURES.
    Reason: 0.99 correlation with AVG_DOWNHOLE_PRESSURE means it carries
    zero additional information. Its high daily noise (7.68 bar/day std)
    combined with covariate shift between the 2008 baseline and 2010 drift
    window caused it to permanently flag as anomalous in ALL scenarios,
    overwhelming the env feature vote and misclassifying every window as
    Environmental Drift regardless of what was actually injected.
    ENV_FEATURES is now 4 features. SENSOR_FEATURES unchanged (1 feature).
    Total model input: 5 features.
"""

import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

RAW_FILE   = "data\Volve production data.xlsx"
SHEET      = "Daily Production Data"
WELL_NAME  = "15/9-F-12"
OUT_FILE   = "f12_clean.csv"
STATS_FILE = "f12_clean_stats.txt"

# CHANGED: AVG_DP_TUBING removed.
# ENV_FEATURES now contains 4 features instead of 5.
ENV_FEATURES = [
    "AVG_DOWNHOLE_PRESSURE",     # reservoir pressure (downhole gauge, ~2km depth)
    "AVG_DOWNHOLE_TEMPERATURE",  # reservoir temperature (same gauge as pressure)
    "BORE_OIL_VOL",              # daily oil production volume at separator
    "AVG_WHP_P",                 # wellhead pressure at surface
]

SENSOR_FEATURES = [
    "AVG_ANNULUS_PRESS",         # annulus pressure — physically isolated from reservoir flow
                                 # detrended correlation with env features: -0.09 to +0.15
                                 # only feature that can deviate independently of the reservoir
]

MIN_ONSTREAM_HRS = 20
BASELINE_FRAC    = 0.30
DRIFT_FRAC       = 0.20

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

print("=" * 60)
print("SAINT Data Preparation — Well 15/9-F-12")
print("=" * 60)

print(f"\n[1] Loading {RAW_FILE} ...")
df_raw = pd.read_excel(RAW_FILE, sheet_name=SHEET, parse_dates=["DATEPRD"])
print(f"    Raw dataset: {len(df_raw):,} rows across all wells and types")

# ── STEP 2: FILTER TO PRODUCER WELLS ONLY ─────────────────────────────────────

producers = df_raw[df_raw["WELL_TYPE"] == "OP"].copy()
print(f"\n[2] Filtered to producer wells (WELL_TYPE='OP'): {len(producers):,} rows")
print(f"    Wells present: {sorted(producers['NPD_WELL_BORE_NAME'].unique())}")

# ── STEP 3: ISOLATE WELL F-12 ─────────────────────────────────────────────────

well = (producers[producers["NPD_WELL_BORE_NAME"] == WELL_NAME]
        .sort_values("DATEPRD")
        .reset_index(drop=True))

print(f"\n[3] Isolated well {WELL_NAME}: {len(well):,} rows")
print(f"    Date range: {well['DATEPRD'].min().date()} → {well['DATEPRD'].max().date()}")

keep_cols = ["DATEPRD", "ON_STREAM_HRS"] + ENV_FEATURES + SENSOR_FEATURES
well = well[keep_cols].copy()

# ── STEP 4: REMOVE SHUTDOWN DAYS ──────────────────────────────────────────────

n_before = len(well)
well = well[well["ON_STREAM_HRS"] >= MIN_ONSTREAM_HRS].copy()
n_shutdown = n_before - len(well)
print(f"\n[4] Removed shutdown days (ON_STREAM_HRS < {MIN_ONSTREAM_HRS}h): "
      f"{n_shutdown} rows removed → {len(well):,} remain")

# ── STEP 5: REMOVE DOWNHOLE GAUGE-PULL DAYS ───────────────────────────────────
# AVG_DOWNHOLE_PRESSURE reads 0.0 when the permanent downhole gauge (PDG)
# is physically removed for maintenance. BORE_OIL_VOL is non-zero on these
# days confirming the well is producing — the sensor is simply absent.

n_before = len(well)
gauge_pull_mask = well["AVG_DOWNHOLE_PRESSURE"] == 0
well = well[~gauge_pull_mask].copy()
n_gauge = n_before - len(well)
print(f"\n[5] Removed downhole gauge-pull days (DH pressure = 0 but well producing): "
      f"{n_gauge} rows removed → {len(well):,} remain")

# ── STEP 6: HANDLE ANNULUS PRESSURE NULLS ─────────────────────────────────────
# 13 consecutive nulls in Nov 2009 — annulus gauge was offline.
# Forward-filled because annulus pressure is very stable (std=5.85 over 8 years)
# and adjacent readings are ~16-17 bar. Flagged in ANNULUS_FILLED column.

well = well.sort_values("DATEPRD").reset_index(drop=True)

annulus_null_mask = well["AVG_ANNULUS_PRESS"].isnull()
n_annulus_nulls   = annulus_null_mask.sum()

well["ANNULUS_FILLED"]   = annulus_null_mask.astype(int)
well["AVG_ANNULUS_PRESS"] = well["AVG_ANNULUS_PRESS"].ffill()

print(f"\n[6] Forward-filled {n_annulus_nulls} annulus pressure nulls (Nov 2009 sensor offline)")
print(f"    Fill value ≈ {well.loc[well['ANNULUS_FILLED']==1, 'AVG_ANNULUS_PRESS'].mean():.2f} bar")

# ── STEP 7: CLIP TEMPERATURE OUTLIERS ─────────────────────────────────────────
# 18 rows with single-day temperature spikes (sensor glitches, not drift).
# Clipped to p01–p99 rather than dropped to preserve the other features on
# those days.

col = "AVG_DOWNHOLE_TEMPERATURE"
q01 = well[col].quantile(0.01)
q99 = well[col].quantile(0.99)
n_clipped = ((well[col] < q01) | (well[col] > q99)).sum()
well[col] = well[col].clip(lower=q01, upper=q99)

print(f"\n[7] Clipped {n_clipped} temperature outliers to "
      f"[p01={q01:.1f}°C, p99={q99:.1f}°C]")

# ── STEP 8: FINAL NULL CHECK ───────────────────────────────────────────────────

all_model_features = ENV_FEATURES + SENSOR_FEATURES
remaining_nulls = well[all_model_features].isnull().sum()
print(f"\n[8] Final null check:")
if remaining_nulls.sum() == 0:
    print("    No nulls remain in model features.")
else:
    print(remaining_nulls[remaining_nulls > 0])

n_before = len(well)
well = well.dropna(subset=all_model_features).reset_index(drop=True)
if n_before - len(well) > 0:
    print(f"    Dropped {n_before - len(well)} rows with remaining nulls")

# ── STEP 9: ASSIGN TEMPORAL WINDOW LABELS ─────────────────────────────────────

n = len(well)
baseline_end = int(n * BASELINE_FRAC)
drift_start  = int(n * (1 - DRIFT_FRAC))

well["WINDOW"] = "normal"
well.loc[:baseline_end - 1, "WINDOW"] = "baseline"
well.loc[drift_start:,       "WINDOW"] = "drift"

print(f"\n[9] Window assignment ({n} total clean rows):")
for w_name in ["baseline", "normal", "drift"]:
    count    = (well["WINDOW"] == w_name).sum()
    date_min = well[well["WINDOW"] == w_name]["DATEPRD"].min().date()
    date_max = well[well["WINDOW"] == w_name]["DATEPRD"].max().date()
    print(f"    {w_name:<10} {count:3d} rows  ({date_min} → {date_max})")

# ── STEP 10: SUMMARY STATISTICS ───────────────────────────────────────────────

print(f"\n[10] Final dataset summary:")
print(f"     Shape: {well.shape}")
print(f"     Features: {all_model_features}")
print(f"\n     Feature statistics:")
print(well[all_model_features].describe().round(3).to_string())

print(f"\n     Correlation matrix (env features — should all be high):")
print(well[ENV_FEATURES].corr().round(3).to_string())

print(f"\n     Annulus vs env correlations (should all be near zero after detrending):")
for f in ENV_FEATURES:
    r = well["AVG_ANNULUS_PRESS"].corr(well[f])
    print(f"     AVG_ANNULUS_PRESS vs {f:<30} r = {r:+.3f}")

# ── STEP 11: SAVE ─────────────────────────────────────────────────────────────

well_out = well.drop(columns=["ON_STREAM_HRS"])
well_out.to_csv(OUT_FILE, index=False)
print(f"\n[11] Saved → {OUT_FILE}  ({len(well_out)} rows)")
print(f"     Columns: {list(well_out.columns)}")

with open(STATS_FILE, "w") as f:
    f.write("SAINT F-12 Data Preparation Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Source file : {RAW_FILE}\n")
    f.write(f"Well        : {WELL_NAME}\n")
    f.write(f"Output file : {OUT_FILE}\n\n")
    f.write("Cleaning steps:\n")
    f.write(f"  Removed {n_shutdown} shutdown days\n")
    f.write(f"  Removed {n_gauge} gauge-pull days\n")
    f.write(f"  Forward-filled {n_annulus_nulls} annulus nulls\n")
    f.write(f"  Clipped {n_clipped} temperature outliers\n\n")
    f.write(f"Final rows  : {len(well_out)}\n")
    f.write(f"Baseline    : {(well['WINDOW']=='baseline').sum()} rows\n")
    f.write(f"Normal      : {(well['WINDOW']=='normal').sum()} rows\n")
    f.write(f"Drift       : {(well['WINDOW']=='drift').sum()} rows\n\n")
    f.write(f"ENV_FEATURES    : {ENV_FEATURES}\n")
    f.write(f"SENSOR_FEATURES : {SENSOR_FEATURES}\n")

print(f"     Saved report → {STATS_FILE}")
print("\nDone. Next step: python inject_drift.py")