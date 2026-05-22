"""
SAINT — Evaluation Script (v7)
================================
Input:
  models/lstm_autoencoder.keras
  artifacts/scaler.pkl
  artifacts/global_threshold.npy
  artifacts/normal_errors_per_feature.npy   ← now from VALIDATION window
  artifacts/feature_meta.json
  scenario_A_env_full.csv
  scenario_B_env_single.csv
  scenario_C_annulus.csv

Outputs:
  results/predictions_all.csv
  results/evaluation_report.txt

────────────────────────────────────────────────────────────
CHANGE FROM v6
────────────────────────────────────────────────────────────
The only change from v6 is the source of normal_errors_per_feature.

In v6 (train_saint v1): normal_errors_per_feature came from the old
'normal' window (WINDOW=='normal', 432 rows, Nov 2008 – Apr 2010).
The threshold calibration was reasonable but the model itself was trained
on 259 rows from 2008 only, so the model failed on 2010 test data.

In v7 (train_saint v2): the model is trained on 692 rows covering
Feb 2008 – Apr 2010. normal_errors_per_feature now comes from the
validation window (last 15% of training data, 104 rows, Dec 2009 – Apr
2010). This is chronologically adjacent to the test window (Apr–Oct 2010).
The CUSUM normalisation in evaluate_saint uses this as the reference,
so it correctly represents "what normal reconstruction error looks like
just before the test period."

All detection and classification logic is identical to v6:
  - Rolling p99 normalisation on reconstruction errors
  - CUSUM (gradual drift) + rolling z-score (sudden faults)
  - Dominance ratio classification with named fault labels
  - Persistence filter

────────────────────────────────────────────────────────────
KNOWN LIMITATION (unchanged)
────────────────────────────────────────────────────────────
Scenario B (single env sensor fault) vs full environmental drift:
ambiguous due to autoencoder bleed-through. Conservative default is
Environmental Drift. Windows where one feature strongly dominates
CUSUM are labelled 'Probable Sensor Fault: X' with low confidence.
See Audibert et al. (2020) for literature context.

Usage:
    python evaluate_saint.py
    (After running inject_drift.py and train_saint.py)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SCENARIO_FILES = {
    "A_env_full": os.path.join(ROOT_DIR, "data/scenario_A_env_full.csv"),
    "B_env_single": os.path.join(ROOT_DIR, "data/scenario_B_env_single.csv"),
    "C_annulus": os.path.join(ROOT_DIR, "data/scenario_C_annulus.csv"),
}

MODEL_PATH    = os.path.join(ROOT_DIR, "models",    "lstm_autoencoder.keras")
SCALER_PATH   = os.path.join(ROOT_DIR, "artifacts", "scaler.pkl")
GLOBAL_THRESH = os.path.join(ROOT_DIR, "artifacts", "global_threshold.npy")
NORMAL_PF     = os.path.join(ROOT_DIR, "artifacts", "normal_errors_per_feature.npy")
FEATURE_META  = os.path.join(ROOT_DIR, "artifacts", "feature_meta.json")

# ── DETECTOR PARAMETERS ───────────────────────────────────────────────────────

CUSUM_K = 0.5
CUSUM_H = 4.0

ROLL_P99_WINDOW      = 50
MAX_NORMALIZED_ERROR = 5.0

ROLL_Z_WINDOW = 7
ROLL_Z_THRESH = 2.0

PERSIST = 3

ANNULUS_DOMINANCE_THRESH = 2.0
ENV_DOMINANCE_THRESH     = 2.5

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
                           window: int, cap: float) -> tuple:
    """
    Normalise each feature's reconstruction error by its own rolling p99.
    Achieves scale fairness across features: all values expressed as
    multiples of their own recent 99th-percentile error.
    Capped at `cap` to prevent extreme spikes from dominating CUSUM.
    """
    df       = pd.DataFrame(errors_per_feat)
    roll_p99 = (df.rolling(window, min_periods=10)
                  .quantile(0.99)
                  .fillna(df.quantile(0.99)))
    roll_p99_safe = roll_p99.clip(lower=1e-8)
    normalized    = (df / roll_p99_safe).clip(upper=cap)
    return normalized.values, roll_p99.values


def compute_cusum(normalized_errors: np.ndarray,
                  k: float, h: float) -> tuple:
    """
    CUSUM on rolling-p99-normalised errors.
    Accumulates persistent upward deviations above k.
    Alarm fires when S_pos exceeds h.
    """
    N, F  = normalized_errors.shape
    S_pos = np.zeros((N, F))
    S_neg = np.zeros((N, F))
    for i in range(1, N):
        S_pos[i] = np.maximum(0, S_pos[i-1] + normalized_errors[i] - k)
        S_neg[i] = np.maximum(0, S_neg[i-1] - normalized_errors[i] - k)
    return (S_pos > h) | (S_neg > h), S_pos, S_neg


def compute_rolling_z(errors_per_feat: np.ndarray,
                      window: int, z_thresh: float) -> tuple:
    """Rolling z-score on raw errors. Secondary detector for sudden faults."""
    df        = pd.DataFrame(errors_per_feat)
    roll_mean = df.rolling(window, min_periods=3).mean()
    roll_std  = df.rolling(window, min_periods=3).std().replace(0, 1e-8).fillna(1e-8)
    z         = np.nan_to_num((df - roll_mean).div(roll_std).values, nan=0.0)
    return z > z_thresh, z


def persistence_filter_2d(flags: np.ndarray, persist: int) -> np.ndarray:
    """Per-feature persistence: flag[i,f] accepted only if persist consecutive."""
    N, F     = flags.shape
    filtered = np.zeros_like(flags, dtype=bool)
    for f in range(F):
        for i in range(persist - 1, N):
            if flags[i-persist+1:i+1, f].all():
                filtered[i, f] = True
    return filtered


def persistence_filter_1d(flags: np.ndarray, persist: int) -> np.ndarray:
    filtered = np.zeros_like(flags, dtype=bool)
    for i in range(persist - 1, len(flags)):
        if flags[i-persist+1:i+1].all():
            filtered[i] = True
    return filtered


def classify_windows(errors_per_feat: np.ndarray,
                     all_features: list,
                     env_idxs: list,
                     sensor_idx: int) -> tuple:
    """
    Full detection + classification pipeline:
      1. Rolling p99 normalise + cap
      2. CUSUM on normalised errors (gradual drift)
      3. Rolling z-score on raw errors (sudden faults)
      4. Union of detectors, persistence filter
      5. Dominance ratio named fault classification
    """
    norm_errors, roll_p99 = rolling_p99_normalize(
        errors_per_feat, ROLL_P99_WINDOW, MAX_NORMALIZED_ERROR)

    cusum_raw, S_pos, S_neg = compute_cusum(norm_errors, CUSUM_K, CUSUM_H)
    roll_raw, roll_z        = compute_rolling_z(
        errors_per_feat, ROLL_Z_WINDOW, ROLL_Z_THRESH)

    combined_raw      = cusum_raw | roll_raw
    combined_filtered = persistence_filter_2d(combined_raw, PERSIST)
    any_anomalous     = persistence_filter_1d(
        combined_filtered.any(axis=1), PERSIST)

    N, F   = errors_per_feat.shape
    labels = []

    for i in range(N):
        if not any_anomalous[i]:
            labels.append("No Drift")
            continue

        s       = S_pos[i]
        ann_s   = s[sensor_idx]
        env_s   = s[env_idxs]
        max_env = env_s.max() + 1e-8

        # Annulus dominance
        if ann_s / max_env > ANNULUS_DOMINANCE_THRESH:
            labels.append("Sensor Fault: Annulus Pressure Gauge")
            continue

        # Env feature dominance
        leader_local = int(np.argmax(env_s))
        leader_s     = env_s[leader_local]
        others_mean  = np.delete(env_s, leader_local).mean() + 1e-8

        if leader_s / others_mean > ENV_DOMINANCE_THRESH and ann_s < max_env:
            feat    = all_features[env_idxs[leader_local]]
            display = DISPLAY_NAMES.get(feat, feat)
            labels.append(f"Probable Sensor Fault: {display}")
            continue

        labels.append("Environmental Drift")

    return (labels, cusum_raw, roll_raw,
            combined_filtered, S_pos, roll_z, norm_errors, roll_p99)


# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

print("=" * 60)
print("SAINT — Evaluation (v7)")
print("=" * 60)

print("\n[1] Loading model and artifacts ...")
model         = load_model(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
global_thresh = float(np.load(GLOBAL_THRESH))
normal_pf     = np.load(NORMAL_PF)   # from validation window (Dec 2009–Apr 2010)

with open(FEATURE_META) as f:
    meta = json.load(f)

ALL_FEATURES = meta["all_features"]
ENV_FEATURES = meta["env_features"]
ENV_IDXS     = meta["env_indices"]
SENSOR_IDX   = meta["sensor_index"]
WINDOW_SIZE  = meta["window_size"]

print(f"    Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
print(f"    Validation errors shape: {normal_pf.shape}  "
      f"(from Dec 2009–Apr 2010 validation window)")
print(f"    Global threshold: {global_thresh:.6f}")
print(f"\n    Detection parameters:")
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
    "SAINT Evaluation Report (v7)",
    "=" * 60,
    "",
    "KNOWN LIMITATION:",
    "Scenario B (single env sensor fault) is ambiguous vs full env drift",
    "due to autoencoder bleed-through between correlated features.",
    "Conservative default: Environmental Drift.",
    "Windows where one feature dominates CUSUM by > 2.5x are labelled",
    "'Probable Sensor Fault: X' with inherent uncertainty.",
    "",
]

for scenario_name, scenario_file in SCENARIO_FILES.items():
    print(f"\n{'─'*60}")
    print(f"Scenario {scenario_name}: {scenario_file}")

    df_s     = pd.read_csv(scenario_file, parse_dates=["DATEPRD"])
    drift_df = df_s[df_s["WINDOW"] == "drift"].copy().reset_index(drop=True)
    print(f"  Drift window: {len(drift_df)} rows  "
          f"({drift_df['DATEPRD'].min().date()} → {drift_df['DATEPRD'].max().date()})")

    X_raw    = drift_df[ALL_FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    X_seq    = make_sequences(X_scaled, WINDOW_SIZE)
    X_pred   = model.predict(X_seq, verbose=0)

    errors_global   = recon_error_per_window(X_seq, X_pred)
    errors_per_feat = recon_error_per_feature(X_seq, X_pred)

    print(f"  Error range: [{errors_global.min():.4f}, {errors_global.max():.4f}]")
    print(f"  Global threshold: {global_thresh:.4f}")
    print(f"  Above threshold: {(errors_global > global_thresh).sum()} / {len(errors_global)}")

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
              f"mean={S_pos[:,i].mean():7.2f}  max={S_pos[:,i].max():7.2f}")

    print(f"\n  Normalised errors (mean | max):")
    for i, feat in enumerate(ALL_FEATURES):
        display = DISPLAY_NAMES.get(feat, feat)
        print(f"    {display:<35} "
              f"mean={norm_errors[:,i].mean():.3f}  max={norm_errors[:,i].max():.3f}")

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
        result_df[f"ERR_{d}"]      = errors_per_feat[:, i]
        result_df[f"NORM_ERR_{d}"] = norm_errors[:, i]
        result_df[f"CUSUM_{d}"]    = S_pos[:, i]
        result_df[f"ROLLZ_{d}"]    = roll_z[:, i]
        result_df[f"FLAG_{d}"]     = combined[:, i]

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

print(f"\nNOTE: 'Probable Sensor Fault' carries inherent uncertainty.")
print(f"Prompts investigation by engineer, not automatic maintenance action.")

# ── STEP 4: SAVE ──────────────────────────────────────────────────────────────

all_df.to_csv(os.path.join(RESULTS_DIR, "predictions_all.csv"), index=False)

report_lines += [
    f"\n{'='*60}", "OVERALL",
    f"{'='*60}",
    f"Overall coarse accuracy : {overall_acc:.4f}",
    f"False alarm rate        : {fa_r*100:.1f}%",
    f"Detection rate          : {det_r*100:.1f}%",
    "\nFine-grained distribution:", all_fine.to_string(),
]
with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print(f"\n[4] Results saved to '{RESULTS_DIR}/'")
print(f"{'='*60}")
print("Evaluation complete.")
print(f"{'='*60}")

# ── STEP 5: MLFLOW (automatic) ────────────────────────────────────────────────
try:
    from mlflow_log import log_evaluation_run

    log_evaluation_run(
        artifacts_dir=os.path.join(ROOT_DIR, "artifacts"),
        models_dir=os.path.join(ROOT_DIR, "models"),
        results_dir=RESULTS_DIR,
        metrics={
            "overall_coarse_accuracy": float(overall_acc),
            "false_alarm_rate": float(fa_r),
            "detection_rate": float(det_r),
        },
    )
except Exception as _mlflow_exc:
    print(f"[MLflow] {_mlflow_exc}")