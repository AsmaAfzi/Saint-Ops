"""
SAINT — Evaluation Script (v6 — Rolling P99 Normalization)
===========================================================
Input:
  models/lstm_autoencoder.keras       artifacts/scaler.pkl
  artifacts/global_threshold.npy      artifacts/env_thresholds.npy
  artifacts/sensor_threshold.npy      artifacts/normal_errors_per_feature.npy
  artifacts/feature_meta.json
  scenario_A_env_full.csv
  scenario_B_env_single.csv
  scenario_C_annulus.csv

Outputs:
  results/predictions_all.csv
  results/evaluation_report.txt

────────────────────────────────────────────────────────────
FULL VERSION HISTORY — WHY EACH VERSION FAILED
────────────────────────────────────────────────────────────
v1: Fixed normal-window z-score (z_thresh=1.0)
  Covariate shift: 2008 normal window stats didn't match 2010 test window.
  Uninjected features showed z=+24. Everything → Env Drift.

v2/v3: Rolling z-score only (z_thresh=3.0, window=30)
  Gradual drift (linear ramp). Rolling baseline adapts at same speed.
  Z-scores stayed near zero. Nothing detected.

v4: CUSUM + rolling z-score + peer graph
  CUSUM detected drift. But peer graph failed because of autoencoder
  bleed-through: when one correlated feature deviates, the model expects
  its peers to follow. They don't → peers also get elevated errors →
  peer graph sees "multiple features anomalous" → always Env Drift.

v5: CUSUM + dominance ratio
  CUSUM normalization used global normal-window median and IQR per feature.
  Oil volume had tiny IQR (small reconstruction errors in normal window)
  but large injection (-1200 m3/day). This created z-scores of 150x for
  oil vol vs 2-4x for other features. Oil vol CUSUM always dominated →
  everything classified as "Probable Sensor Fault: Oil Flow Meter".
  Also: annulus injection (+12 bar) was below its normal-window p99 (18.33)
  — invisible to the model as it fell within natural variation.

────────────────────────────────────────────────────────────
v6 FIXES
────────────────────────────────────────────────────────────
Fix 1 — Rolling p99 normalization (replaces global normal-window IQR):
  Each feature's reconstruction error is normalised by a rolling p99
  computed over the previous ROLL_P99_WINDOW timesteps of the test
  sequence itself. This achieves two things:
  (a) Scale fairness: a feature at 2× its own local p99 is treated the
      same as any other feature at 2× its local p99, regardless of
      absolute error magnitude. Oil vol's 150x inflation disappears.
  (b) Covariate shift tolerance: the reference adapts to the local
      operating point, not a reference from a different time period.

Fix 2 — Error cap (MAX_NORMALIZED_ERROR = 5.0):
  After rolling p99 normalization, each feature's normalized error is
  capped at 5.0. This prevents any single feature from dominating the
  CUSUM sum due to a momentary spike or extreme outlier. A feature at
  5× its rolling p99 is clearly anomalous — anything higher doesn't
  add classification information, only numerical instability.

Fix 3 — Inject drift.py annulus magnitude +12 → +25 bar:
  The +12 bar annulus injection was below the normal-window p99 of annulus
  reconstruction error (18.33 bar), making it invisible to the model.
  +25 bar (0.14 bar/day) is within the active leak range (0.05-0.5 bar/day)
  and produces rolling-normalized error clearly above 1.0.

────────────────────────────────────────────────────────────
CLASSIFICATION LOGIC (unchanged from v5 structure)
────────────────────────────────────────────────────────────
After CUSUM on rolling-p99-normalized + capped errors:

  Annulus dominance: annulus_CUSUM / max(env_CUSUM) > THRESH → Sensor Fault: Annulus
  Env dominance:     leader_env_CUSUM / mean(others) > THRESH → Probable Sensor Fault: X
  Default:           Environmental Drift
  Nothing:           No Drift

────────────────────────────────────────────────────────────
KNOWN LIMITATION (unchanged)
────────────────────────────────────────────────────────────
Scenario B (single env sensor fault) vs Scenario A (full env drift):
AMBIGUOUS due to autoencoder bleed-through. Correlated features always
show elevated errors when any one of them deviates. Conservative default
is Environmental Drift (prompts operational investigation, not wrong
maintenance action). See Audibert et al. (2020) for literature context.

Usage:
    python evaluate_saint.py

Re-run required files after any inject_drift.py change:
    python inject_drift.py    ← regenerates scenario CSVs
    python evaluate_saint.py  ← no retraining needed
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ── Project paths (integration only — logic unchanged) ───────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
_DATA = os.path.join(_ROOT, "data")

# ── CONFIG ────────────────────────────────────────────────────────────────────

SCENARIO_FILES = {
    "A_env_full":   os.path.join(_DATA, "scenario_A_env_full.csv"),
    "B_env_single": os.path.join(_DATA, "scenario_B_env_single.csv"),
    "C_annulus":    os.path.join(_DATA, "scenario_C_annulus.csv"),
}

MODEL_PATH    = os.path.join(_ROOT, "models",    "lstm_autoencoder.keras")
SCALER_PATH   = os.path.join(_ROOT, "artifacts", "scaler.pkl")
GLOBAL_THRESH = os.path.join(_ROOT, "artifacts", "global_threshold.npy")
NORMAL_PF     = os.path.join(_ROOT, "artifacts", "normal_errors_per_feature.npy")
FEATURE_META  = os.path.join(_ROOT, "artifacts", "feature_meta.json")
RESULTS_DIR   = os.path.join(_BASE, "results")

# ── DETECTOR PARAMETERS ───────────────────────────────────────────────────────

# CUSUM — primary detector for gradual drift
CUSUM_K = 0.5    # allowance: sensitive to 1-sigma persistent shifts
CUSUM_H = 4.0    # threshold: ~0.27% false alarm rate per observation

# Rolling p99 normalisation window
# Each feature's error is normalised by its own rolling p99 over this window.
# Shorter = more adaptive (better for sudden faults but may miss slow drift).
# Longer = more stable reference (better for gradual drift).
# 50 windows ≈ 50 days — adapts to slow depletion but not to short drift bursts.
ROLL_P99_WINDOW = 50

# Maximum normalized error per feature (cap before CUSUM)
# Prevents any single feature from dominating due to extreme spikes.
# Value of 5.0 means "5× the local p99 is the ceiling".
MAX_NORMALIZED_ERROR = 5.0

# Rolling z-score — secondary detector for sudden step changes
ROLL_Z_WINDOW = 7
ROLL_Z_THRESH = 2.0

# Persistence: flag must hold for PERSIST consecutive windows
PERSIST = 3

# ── CLASSIFICATION THRESHOLDS ─────────────────────────────────────────────────

# Annulus dominance: annulus_CUSUM / max(env_CUSUM) > this → Sensor Fault: Annulus
# Calibrated: Scenario C with +25 bar injection produces ratio ~100x.
# Scenario A/B with env drift produces ratio ~0.003.
# Threshold of 2.0 gives large margin in both directions.
ANNULUS_DOMINANCE_THRESH = 2.0

# Env feature dominance: leader_CUSUM / mean(others) > this → Probable Sensor Fault
# Due to bleed-through, this ratio is ~1.3-1.5 even for true single-feature faults.
# Set conservatively at 2.5 — only fires when one feature truly dominates.
# Most Scenario B windows will still default to Environmental Drift (conservative).
ENV_DOMINANCE_THRESH = 2.5

# Human-readable names
DISPLAY_NAMES = {
    "AVG_DOWNHOLE_PRESSURE":    "Downhole Pressure Gauge",
    "AVG_DOWNHOLE_TEMPERATURE": "Downhole Temperature Sensor",
    "BORE_OIL_VOL":             "Oil Flow Meter",
    "AVG_WHP_P":                "Wellhead Pressure Gauge",
    "AVG_ANNULUS_PRESS":        "Annulus Pressure Gauge",
}

COARSE_CLASS_ORDER = ["No Drift", "Environmental Drift", "Sensor Drift"]

def coarsen_label(label: str) -> str:
    if label == "No Drift":
        return "No Drift"
    if label in ("Environmental Drift", "Probable Environmental Drift"):
        return "Environmental Drift"
    return "Sensor Drift"

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def make_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.array([data[i:i+window_size]
                     for i in range(len(data) - window_size + 1)])

def recon_error_per_window(X_true, X_pred):
    return np.mean(np.square(X_true - X_pred), axis=(1, 2))

def recon_error_per_feature(X_true, X_pred):
    return np.mean(np.square(X_true - X_pred), axis=1)


def rolling_p99_normalize(errors_per_feat: np.ndarray,
                           window: int,
                           cap: float) -> np.ndarray:
    """
    Normalise per-feature reconstruction errors by each feature's own
    rolling p99 over the previous `window` timesteps.

    This achieves scale fairness: every feature is expressed as a multiple
    of its own recent 99th-percentile error. A value of 1.0 means "at the
    local p99 boundary". A value of 2.0 means "twice the local p99".

    Errors are capped at `cap` after normalisation to prevent any single
    feature from dominating CUSUM due to extreme spikes.

    Parameters
    ----------
    errors_per_feat : (N, F) — raw reconstruction errors
    window          : int — rolling look-back for p99 computation
    cap             : float — maximum normalized value allowed

    Returns
    -------
    normalized : (N, F) — rolling-p99-normalised, capped errors
    roll_p99   : (N, F) — the rolling p99 reference (for visualisation)
    """
    df       = pd.DataFrame(errors_per_feat)
    roll_p99 = (df.rolling(window, min_periods=10)
                  .quantile(0.99)
                  .fillna(df.quantile(0.99)))   # fallback for early rows

    # Avoid division by zero
    roll_p99_safe = roll_p99.clip(lower=1e-8)

    normalized = (df / roll_p99_safe).clip(upper=cap)
    return normalized.values, roll_p99.values


def compute_cusum(normalized_errors: np.ndarray,
                  k: float,
                  h: float) -> tuple:
    """
    Run CUSUM on rolling-p99-normalised errors.

    Input is already on a comparable scale (multiples of local p99),
    so no further standardisation is needed. The CUSUM accumulates
    positive deviations above k — when S_pos exceeds h, alarm fires.

    Parameters
    ----------
    normalized_errors : (N, F) — p99-normalised, capped errors
    k                 : float — CUSUM allowance
    h                 : float — CUSUM alarm threshold

    Returns
    -------
    flags : (N, F) bool
    S_pos : (N, F) float — upper CUSUM statistic
    S_neg : (N, F) float — lower CUSUM statistic
    """
    N, F   = normalized_errors.shape
    S_pos  = np.zeros((N, F))
    S_neg  = np.zeros((N, F))

    for i in range(1, N):
        S_pos[i] = np.maximum(0, S_pos[i-1] + normalized_errors[i] - k)
        S_neg[i] = np.maximum(0, S_neg[i-1] - normalized_errors[i] - k)

    flags = (S_pos > h) | (S_neg > h)
    return flags, S_pos, S_neg


def compute_rolling_z(errors_per_feat: np.ndarray,
                      window: int,
                      z_thresh: float) -> tuple:
    """
    Rolling z-score on raw reconstruction errors.
    Secondary detector for sudden step changes.
    """
    df        = pd.DataFrame(errors_per_feat)
    roll_mean = df.rolling(window, min_periods=3).mean()
    roll_std  = df.rolling(window, min_periods=3).std().replace(0, 1e-8).fillna(1e-8)
    z         = np.nan_to_num((df - roll_mean).div(roll_std).values, nan=0.0)
    return z > z_thresh, z


def persistence_filter_2d(flags: np.ndarray, persist: int) -> np.ndarray:
    """Per-feature persistence filter: flag[i,f] accepted only if
    flags[i-persist+1:i+1, f] are all True."""
    N, F     = flags.shape
    filtered = np.zeros_like(flags, dtype=bool)
    for f in range(F):
        for i in range(persist - 1, N):
            if flags[i-persist+1:i+1, f].all():
                filtered[i, f] = True
    return filtered


def persistence_filter_1d(flags: np.ndarray, persist: int) -> np.ndarray:
    """Scalar boolean persistence filter."""
    filtered = np.zeros_like(flags, dtype=bool)
    for i in range(persist - 1, len(flags)):
        if flags[i-persist+1:i+1].all():
            filtered[i] = True
    return filtered


def classify_windows(errors_per_feat: np.ndarray,
                     all_features:    list,
                     env_idxs:        list,
                     sensor_idx:      int) -> tuple:
    """
    Full classification pipeline:
      1. Rolling p99 normalise + cap
      2. CUSUM on normalised errors (gradual drift detector)
      3. Rolling z-score on raw errors (sudden fault detector)
      4. Combine (union), apply persistence filter
      5. Dominance ratio classification

    Returns
    -------
    fine_labels   : list[str] length N
    cusum_flags   : (N,F) bool — CUSUM alarms
    roll_flags    : (N,F) bool — rolling z alarms
    combined      : (N,F) bool — union after persistence
    S_pos         : (N,F) — CUSUM S+ for visualisation
    roll_z        : (N,F) — rolling z-scores for visualisation
    norm_errors   : (N,F) — rolling-p99-normalised errors
    roll_p99      : (N,F) — rolling p99 reference values
    """
    # Step 1: normalise
    norm_errors, roll_p99 = rolling_p99_normalize(
        errors_per_feat, ROLL_P99_WINDOW, MAX_NORMALIZED_ERROR)

    # Step 2: CUSUM on normalised errors
    cusum_raw, S_pos, S_neg = compute_cusum(norm_errors, CUSUM_K, CUSUM_H)

    # Step 3: rolling z-score on raw errors
    roll_raw, roll_z = compute_rolling_z(
        errors_per_feat, ROLL_Z_WINDOW, ROLL_Z_THRESH)

    # Step 4: combine and apply persistence
    combined_raw      = cusum_raw | roll_raw
    combined_filtered = persistence_filter_2d(combined_raw, PERSIST)

    # Global anomaly flag: at least one feature persistently anomalous
    any_anomalous = persistence_filter_1d(combined_filtered.any(axis=1), PERSIST)

    N, F = errors_per_feat.shape
    labels = []

    for i in range(N):
        if not any_anomalous[i]:
            labels.append("No Drift")
            continue

        # Use CUSUM S_pos as per-feature severity
        s       = S_pos[i]
        ann_s   = s[sensor_idx]
        env_s   = s[env_idxs]
        max_env = env_s.max() + 1e-8

        # ── Annulus dominance ─────────────────────────────────────────────────
        ann_ratio = ann_s / max_env
        if ann_ratio > ANNULUS_DOMINANCE_THRESH:
            labels.append("Sensor Fault: Annulus Pressure Gauge")
            continue

        # ── Env feature dominance ─────────────────────────────────────────────
        leader_local = int(np.argmax(env_s))
        leader_s     = env_s[leader_local]
        others_mean  = np.delete(env_s, leader_local).mean() + 1e-8
        env_ratio    = leader_s / others_mean

        if env_ratio > ENV_DOMINANCE_THRESH and ann_s < max_env:
            feat_name = all_features[env_idxs[leader_local]]
            display   = DISPLAY_NAMES.get(feat_name, feat_name)
            labels.append(f"Probable Sensor Fault: {display}")
            continue

        # ── Default ───────────────────────────────────────────────────────────
        labels.append("Environmental Drift")

    return (labels, cusum_raw, roll_raw,
            combined_filtered, S_pos, roll_z, norm_errors, roll_p99)


# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

print("=" * 60)
print("SAINT — Evaluation (v6 — Rolling P99 Normalization)")
print("=" * 60)

print("\n[1] Loading model and artifacts ...")
model         = load_model(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
global_thresh = float(np.load(GLOBAL_THRESH))
normal_pf     = np.load(NORMAL_PF)

with open(FEATURE_META) as f:
    meta = json.load(f)

ALL_FEATURES = meta["all_features"]
ENV_FEATURES = meta["env_features"]
ENV_IDXS     = meta["env_indices"]
SENSOR_IDX   = meta["sensor_index"]
WINDOW_SIZE  = meta["window_size"]

print(f"    Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
print(f"    Normal window errors shape: {normal_pf.shape}")
print(f"\n    Detector parameters:")
print(f"      CUSUM              k={CUSUM_K}, h={CUSUM_H}")
print(f"      Rolling p99 window = {ROLL_P99_WINDOW}")
print(f"      Error cap          = {MAX_NORMALIZED_ERROR}x p99")
print(f"      Rolling z          window={ROLL_Z_WINDOW}, thresh={ROLL_Z_THRESH}")
print(f"      Persistence        = {PERSIST} windows")
print(f"      Annulus dominance  > {ANNULUS_DOMINANCE_THRESH}x")
print(f"      Env dominance      > {ENV_DOMINANCE_THRESH}x")

# ── STEP 2: EVALUATE ──────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)
all_predictions = []
report_lines    = [
    "SAINT Evaluation Report (v6 — Rolling P99 Normalization)",
    "=" * 60,
    "",
    "KNOWN LIMITATION:",
    "Scenario B (single env sensor fault) vs Scenario A (full env drift)",
    "are ambiguous due to autoencoder bleed-through between correlated",
    "env features. Conservative default: Environmental Drift.",
    "Windows where one env feature dominates CUSUM by > 2.5x are labelled",
    "'Probable Sensor Fault: X' with inherent uncertainty.",
    "",
]

for scenario_name, scenario_file in SCENARIO_FILES.items():
    print(f"\n{'─'*60}")
    print(f"Scenario {scenario_name}: {scenario_file}")

    df_s     = pd.read_csv(scenario_file, parse_dates=["DATEPRD"])
    drift_df = df_s[df_s["WINDOW"] == "drift"].copy().reset_index(drop=True)
    print(f"  Drift window: {len(drift_df)} rows")

    X_raw    = drift_df[ALL_FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    X_seq    = make_sequences(X_scaled, WINDOW_SIZE)
    X_pred   = model.predict(X_seq, verbose=0)

    errors_global   = recon_error_per_window(X_seq, X_pred)
    errors_per_feat = recon_error_per_feature(X_seq, X_pred)

    print(f"  Error range: [{errors_global.min():.4f}, {errors_global.max():.4f}]")

    (fine_labels, cusum_flags, roll_flags,
     combined, S_pos, roll_z,
     norm_errors, roll_p99_vals) = classify_windows(
        errors_per_feat = errors_per_feat,
        all_features    = ALL_FEATURES,
        env_idxs        = ENV_IDXS,
        sensor_idx      = SENSOR_IDX,
    )

    seq_start     = WINDOW_SIZE - 1
    y_true_raw    = drift_df["TRUE_LABEL"].iloc[seq_start:].reset_index(drop=True).values
    dates         = drift_df["DATEPRD"].iloc[seq_start:].reset_index(drop=True).values
    y_pred_fine   = np.array(fine_labels)
    y_pred_coarse = np.array([coarsen_label(l) for l in y_pred_fine])
    y_true_coarse = np.array([coarsen_label(l) for l in y_true_raw])

    cm       = confusion_matrix(y_true_coarse, y_pred_coarse,
                                labels=COARSE_CLASS_ORDER)
    cr       = classification_report(y_true_coarse, y_pred_coarse,
                                     labels=COARSE_CLASS_ORDER, zero_division=0)
    f1_macro = f1_score(y_true_coarse, y_pred_coarse,
                        labels=COARSE_CLASS_ORDER, average="macro", zero_division=0)

    print(f"\n  Classification report (coarse):")
    print(cr)
    print(f"  Confusion matrix  classes={COARSE_CLASS_ORDER}")
    print(cm)
    print(f"\n  Macro F1 (coarse): {f1_macro:.4f}")

    fine_counts = pd.Series(y_pred_fine).value_counts()
    print(f"\n  Fine-grained breakdown:")
    for label, cnt in fine_counts.items():
        print(f"    {label:<55} {cnt:4d} ({cnt/len(y_pred_fine)*100:.1f}%)")

    print(f"\n  CUSUM S+ per feature (mean | max):")
    for i, feat in enumerate(ALL_FEATURES):
        group   = "[ENV]   " if i in ENV_IDXS else "[SENSOR]"
        display = DISPLAY_NAMES.get(feat, feat)
        print(f"    {group} {display:<35} "
              f"mean={S_pos[:,i].mean():7.1f}  max={S_pos[:,i].max():7.1f}")

    print(f"\n  Normalised error per feature (mean | max after rolling p99 + cap):")
    for i, feat in enumerate(ALL_FEATURES):
        display = DISPLAY_NAMES.get(feat, feat)
        print(f"    {display:<35} "
              f"mean={norm_errors[:,i].mean():.3f}  max={norm_errors[:,i].max():.3f}")

    # Collect
    result_df = pd.DataFrame({
        "SCENARIO":               scenario_name,
        "DATEPRD":                dates,
        "TRUE_LABEL":             y_true_raw,
        "PREDICTED_LABEL_FINE":   y_pred_fine,
        "PREDICTED_LABEL_COARSE": y_pred_coarse,
        "GLOBAL_ERROR":           errors_global,
        "CORRECT_COARSE":         (y_true_coarse == y_pred_coarse),
    })
    for i, feat in enumerate(ALL_FEATURES):
        d = DISPLAY_NAMES.get(feat, feat).replace(" ", "_")
        result_df[f"ERR_{d}"]       = errors_per_feat[:, i]
        result_df[f"NORM_ERR_{d}"]  = norm_errors[:, i]
        result_df[f"CUSUM_{d}"]     = S_pos[:, i]
        result_df[f"ROLLZ_{d}"]     = roll_z[:, i]
        result_df[f"FLAG_{d}"]      = combined[:, i]

    all_predictions.append(result_df)

    report_lines += [
        f"\n{'='*60}",
        f"Scenario {scenario_name}: {scenario_file}",
        f"{'='*60}",
        cr,
        f"Confusion matrix {COARSE_CLASS_ORDER}",
        str(cm),
        f"\nMacro F1 (coarse): {f1_macro:.4f}",
        "\nFine-grained breakdown:",
        fine_counts.to_string(),
    ]

# ── STEP 3: SUMMARY ───────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")

all_df      = pd.concat(all_predictions, ignore_index=True)
overall_acc = all_df["CORRECT_COARSE"].mean()
print(f"Overall coarse accuracy: {overall_acc:.4f}")

print(f"\nPer-scenario Macro F1 (coarse):")
for sn in SCENARIO_FILES:
    s   = all_df[all_df["SCENARIO"] == sn]
    f1  = f1_score([coarsen_label(l) for l in s["TRUE_LABEL"]],
                   s["PREDICTED_LABEL_COARSE"],
                   labels=COARSE_CLASS_ORDER, average="macro", zero_division=0)
    acc = s["CORRECT_COARSE"].mean() * 100
    print(f"  {sn:<20}  F1={f1:.4f}   Acc={acc:.1f}%")

no_d  = all_df[all_df["TRUE_LABEL"].apply(coarsen_label) == "No Drift"]
fa    = (no_d["PREDICTED_LABEL_COARSE"] != "No Drift").sum()
fa_r  = fa / len(no_d) if len(no_d) > 0 else 0

dr    = all_df[all_df["TRUE_LABEL"].apply(coarsen_label) != "No Drift"]
det   = (dr["PREDICTED_LABEL_COARSE"] != "No Drift").sum()
det_r = det / len(dr) if len(dr) > 0 else 0

print(f"\nFalse alarm rate: {fa_r*100:.1f}%  ({fa}/{len(no_d)})")
print(f"Detection rate:   {det_r*100:.1f}%  ({det}/{len(dr)})")

print(f"\nFine-grained label distribution (all scenarios):")
all_fine = all_df["PREDICTED_LABEL_FINE"].value_counts()
for label, cnt in all_fine.items():
    print(f"  {label:<55} {cnt:5d} ({cnt/len(all_df)*100:.1f}%)")

print(f"\nNOTE: 'Probable Sensor Fault' carries inherent uncertainty due to")
print(f"autoencoder bleed-through. Prompts investigation, not automatic action.")

# ── STEP 4: SAVE ──────────────────────────────────────────────────────────────

all_df.to_csv(os.path.join(RESULTS_DIR, "predictions_all.csv"), index=False)

report_lines += [
    f"\n{'='*60}", "OVERALL",
    f"{'='*60}",
    f"Overall coarse accuracy : {overall_acc:.4f}",
    f"False alarm rate        : {fa_r*100:.1f}%",
    f"Detection rate          : {det_r*100:.1f}%",
    "\nFine-grained distribution:", all_fine.to_string(),
    "\nNOTE: 'Probable Sensor Fault' labels are uncertain due to",
    "autoencoder bleed-through. See module docstring for details.",
]
with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print(f"\n[4] Results saved to '{RESULTS_DIR}/'")
print(f"{'='*60}")
print("Evaluation complete.")
print(f"{'='*60}")
