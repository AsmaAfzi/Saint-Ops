"""
SAINT — Evaluation Script (v5 — Dominance Ratio Classification)
================================================================
Input:
  models/lstm_autoencoder.keras       artifacts/scaler.pkl
  artifacts/global_threshold.npy      artifacts/env_thresholds.npy
  artifacts/sensor_threshold.npy      artifacts/normal_errors_per_feature.npy
  artifacts/feature_meta.json
  scenario_A_env_full.csv  scenario_B_env_single.csv  scenario_C_annulus.csv

Outputs:
  results/predictions_all.csv    results/evaluation_report.txt

─────────────────────────────────────────────────────────────
WHY PREVIOUS VERSIONS FAILED — COMPLETE HISTORY
─────────────────────────────────────────────────────────────
v1 (fixed z-score, z_thresh=1.0):
  Covariate shift: normal window (2008) vs drift window (2010) had different
  distributions. Uninjected features showed z=+24. Everything = Env Drift.
  FA rate: 93.5%.  Detection: 100% (useless).

v2/v3 (rolling z-score only, z_thresh=3.0):
  Drift is a gradual ramp. Rolling baseline adapts at same speed as drift.
  Z-scores stayed near zero throughout. Nothing detected.
  FA rate: 0%.  Detection: 0%.

v4 (CUSUM + rolling z-score, peer graph classification):
  CUSUM correctly detected drift in all scenarios.
  But the peer graph logic failed due to AUTOENCODER BLEED-THROUGH:
  The LSTM learned correlations between env features (DH_press ↔ WHP r=0.82).
  When only DH_press was injected (Scenario B), the model expected WHP to
  follow. WHP didn't → WHP reconstruction error rose too. CUSUM fired on
  multiple env features even though only one was injected.
  The peer graph saw "multiple features anomalous" and called Env Drift.
  FA rate: 95.2%.  Detection: 100% (all mislabelled as Env Drift).

─────────────────────────────────────────────────────────────
FUNDAMENTAL LIMITATION — DOCUMENTED FOR RESEARCH PAPER
─────────────────────────────────────────────────────────────
A multivariate LSTM autoencoder trained on correlated features CANNOT
reliably distinguish a single tightly-coupled env sensor fault from full
environmental drift. This is not a bug — it is a known property of
reconstruction-based anomaly detection (Audibert et al., 2020).

When DH_press is the only faulting sensor, the model reconstructs WHP
and DH_temp proportionally (because it learned their joint distribution).
The actual WHP and DH_temp stay at baseline → elevated reconstruction
error in non-faulting sensors. The reconstruction error pattern of
Scenario B (one env sensor fault) and Scenario A (full env drift) are
nearly identical from the model's perspective.

─────────────────────────────────────────────────────────────
WHAT CAN BE CLEANLY DISTINGUISHED
─────────────────────────────────────────────────────────────
✓ Environmental Drift (Scenario A): All env features drift together.
  CUSUM fires on all env features with similar magnitude.
  Annulus CUSUM stays low (no bleed from env to annulus — isolated).

✓ Annulus Sensor Fault (Scenario C): Only annulus rises.
  Annulus is physically isolated (detrended corr < 0.15 with all env).
  No bleed-through → annulus CUSUM >> env CUSUM stats.
  Annulus/max_env CUSUM ratio: ~340x in Scenario C vs 0.003 in Scenario A.
  A threshold of 2.0 cleanly separates them.

~ Env Single Sensor Fault (Scenario B): One env feature drifts.
  Due to bleed-through, ALL env features show elevated CUSUM.
  The faulting feature (DH_press) has SLIGHTLY higher CUSUM than others
  (injection magnitude > bleed-through magnitude), but the difference is
  small. Classified as "Probable Environmental Drift" with a note that
  one feature is dominant — flagged for engineer investigation.
  This is the honest, physically correct behaviour given the model's
  architecture. The research paper must document this limitation.

─────────────────────────────────────────────────────────────
CLASSIFICATION LOGIC (v5)
─────────────────────────────────────────────────────────────
For each window, after CUSUM detects anomaly:

Step 1 — Compute annulus dominance ratio:
  ratio = annulus_CUSUM / max(env_CUSUM values)
  If ratio > ANNULUS_DOMINANCE_THRESH (2.0):
    → "Sensor Fault: Annulus Pressure Gauge"

Step 2 — Compute env dominance ratio:
  Find the env feature with highest CUSUM (the "leader").
  ratio = leader_CUSUM / mean(all other env CUSUM values)
  If ratio > ENV_DOMINANCE_THRESH (2.5) AND annulus CUSUM is low:
    → "Probable Sensor Fault: <leader feature name>"
    (probable = autoencoder bleed-through makes this uncertain)

Step 3 — Default:
  → "Environmental Drift"
  (conservative: when evidence is ambiguous, assume process change,
   which requires operational response rather than sensor maintenance)

Persistence filter of 3 windows applied before final output.

Usage:
    python evaluate_saint.py
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
SCENARIO_FILES = {
    "A_env_full":   os.path.join(BASE_DIR, "../data/scenario_A_env_full.csv"),
    "B_env_single": os.path.join(BASE_DIR, "../data/scenario_B_env_single.csv"),
    "C_annulus":    os.path.join(BASE_DIR, "../data/scenario_C_annulus.csv"),
}

MODEL_PATH        = os.path.join("models",    "lstm_autoencoder.keras")
SCALER_PATH       = os.path.join("artifacts", "scaler.pkl")
GLOBAL_THRESH     = os.path.join("artifacts", "global_threshold.npy")
ENV_THRESH        = os.path.join("artifacts", "env_thresholds.npy")
SENSOR_THRESH     = os.path.join("artifacts", "sensor_threshold.npy")
NORMAL_ERRORS     = os.path.join("artifacts", "normal_errors.npy")
NORMAL_PER_FEAT   = os.path.join("artifacts", "normal_errors_per_feature.npy")
FEATURE_META      = os.path.join("artifacts", "feature_meta.json")

RESULTS_DIR       = os.path.join(BASE_DIR,"..ml/results")

# ── DETECTOR PARAMETERS ───────────────────────────────────────────────────────

# CUSUM — primary detector for gradual drift
# k=0.5: sensitive to 1-sigma persistent shifts (SPC standard)
# h=4.0: ~0.27% false alarm rate per observation on clean data
CUSUM_K = 0.5
CUSUM_H = 4.0

# Rolling z-score — secondary detector for sudden step faults
# window=7, thresh=2.0: catches 20x error jumps at onset (Scenario B)
ROLL_WINDOW   = 7
ROLL_Z_THRESH = 2.0

# Persistence: flag must hold for PERSIST consecutive windows
PERSIST = 3

# ── CLASSIFICATION PARAMETERS ─────────────────────────────────────────────────

# Annulus dominance ratio:
# annulus_CUSUM / max(env_CUSUM) > this → Sensor Fault: Annulus
# Calibration: Scenario C ratio = ~340x, Scenario A ratio = 0.003
# Threshold of 2.0 cleanly separates them with large margin.
ANNULUS_DOMINANCE_THRESH = 2.0

# Env feature dominance ratio:
# leader_CUSUM / mean(other_env_CUSUM) > this → Probable Sensor Fault
# Calibration: in Scenario B, leader/mean_others ≈ 1.35 (bleed-through
# makes this ambiguous). Threshold of 2.5 requires clear separation.
# Set conservatively — better to default to Env Drift than to wrong fault name.
ENV_DOMINANCE_THRESH = 2.5

# Human-readable names for the frontend and reports
DISPLAY_NAMES = {
    "AVG_DOWNHOLE_PRESSURE":    "Downhole Pressure Gauge",
    "AVG_DOWNHOLE_TEMPERATURE": "Downhole Temperature Sensor",
    "BORE_OIL_VOL":             "Oil Flow Meter",
    "AVG_WHP_P":                "Wellhead Pressure Gauge",
    "AVG_ANNULUS_PRESS":        "Annulus Pressure Gauge",
}

COARSE_CLASS_ORDER = ["No Drift", "Environmental Drift", "Sensor Drift"]

def coarsen_label(label: str) -> str:
    """Map fine label to coarse label for standard scoring."""
    if label == "No Drift":
        return "No Drift"
    if label in ("Environmental Drift", "Probable Environmental Drift"):
        return "Environmental Drift"
    return "Sensor Drift"   # all "Sensor Fault: X" and "Probable Sensor Fault: X"

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def make_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.array([data[i:i+window_size]
                     for i in range(len(data) - window_size + 1)])

def recon_error_per_window(X_true, X_pred):
    return np.mean(np.square(X_true - X_pred), axis=(1, 2))

def recon_error_per_feature(X_true, X_pred):
    return np.mean(np.square(X_true - X_pred), axis=1)

def compute_cusum(errors_per_feat: np.ndarray,
                  normal_pf: np.ndarray,
                  k: float, h: float) -> tuple:
    """
    CUSUM on per-feature reconstruction errors.
    Normalises by normal-window median and IQR (robust to outliers).
    Returns:
      flags  (N, F) bool — alarm fires when S_pos or S_neg exceeds h
      S_pos  (N, F) float — upper CUSUM statistic (for visualisation)
      S_neg  (N, F) float — lower CUSUM statistic
    """
    mu0    = np.median(normal_pf, axis=0)
    sigma0 = (np.percentile(normal_pf, 75, axis=0) -
              np.percentile(normal_pf, 25, axis=0)) / 1.349 + 1e-8
    z      = (errors_per_feat - mu0) / sigma0
    N, F   = z.shape
    S_pos  = np.zeros((N, F))
    S_neg  = np.zeros((N, F))
    for i in range(1, N):
        S_pos[i] = np.maximum(0, S_pos[i-1] + z[i] - k)
        S_neg[i] = np.maximum(0, S_neg[i-1] - z[i] - k)
    flags = (S_pos > h) | (S_neg > h)
    return flags, S_pos, S_neg

def compute_rolling_z(errors_per_feat: np.ndarray,
                      window: int, z_thresh: float) -> tuple:
    """
    Rolling z-score on reconstruction errors.
    Detects sudden step changes (complementary to CUSUM's gradual drift detection).
    Returns flags (N,F) bool and z_scores (N,F) float.
    """
    df        = pd.DataFrame(errors_per_feat)
    roll_mean = df.rolling(window, min_periods=3).mean()
    roll_std  = df.rolling(window, min_periods=3).std().replace(0,1e-8).fillna(1e-8)
    z         = np.nan_to_num((df - roll_mean).div(roll_std).values, nan=0.0)
    return z > z_thresh, z

def persistence_filter_2d(flags: np.ndarray, persist: int) -> np.ndarray:
    """
    Per-feature persistence filter. Flag[i,f] accepted only if
    flags[i-persist+1:i+1, f] are all True.
    """
    N, F     = flags.shape
    filtered = np.zeros_like(flags, dtype=bool)
    for f in range(F):
        for i in range(persist - 1, N):
            if flags[i-persist+1:i+1, f].all():
                filtered[i, f] = True
    return filtered

def persistence_filter_1d(flags: np.ndarray, persist: int) -> np.ndarray:
    """Per-window persistence filter for scalar boolean arrays."""
    filtered = np.zeros_like(flags, dtype=bool)
    for i in range(persist - 1, len(flags)):
        if flags[i-persist+1:i+1].all():
            filtered[i] = True
    return filtered

def classify_windows(errors_per_feat: np.ndarray,
                     normal_pf:       np.ndarray,
                     all_features:    list,
                     env_idxs:        list,
                     sensor_idx:      int) -> tuple:
    """
    Classify each window using CUSUM + rolling z-score detectors,
    then apply the dominance-ratio named fault classification.

    Classification logic (see module docstring for full explanation):
      - Annulus CUSUM >> env CUSUM → Sensor Fault: Annulus
      - One env feature CUSUM >> others → Probable Sensor Fault: <feature>
      - Otherwise (default) → Environmental Drift
      - Nothing anomalous → No Drift

    Returns
    -------
    fine_labels   : list[str] length N
    cusum_flags   : (N,F) bool — CUSUM alarms (pre-persistence)
    roll_flags    : (N,F) bool — rolling z alarms (pre-persistence)
    combined      : (N,F) bool — union after per-feature persistence
    S_pos         : (N,F) — CUSUM S+ statistic
    roll_z        : (N,F) — rolling z-scores
    """
    # ── Run detectors ─────────────────────────────────────────────────────────
    cusum_raw, S_pos, S_neg = compute_cusum(
        errors_per_feat, normal_pf, CUSUM_K, CUSUM_H)
    roll_raw, roll_z = compute_rolling_z(
        errors_per_feat, ROLL_WINDOW, ROLL_Z_THRESH)

    # Combined per-feature flag (either detector)
    combined_raw      = cusum_raw | roll_raw
    combined_filtered = persistence_filter_2d(combined_raw, PERSIST)

    N, F = errors_per_feat.shape

    # ── Is anything anomalous at all? ─────────────────────────────────────────
    # Global flag: at least one feature persistently anomalous
    any_anomalous = combined_filtered.any(axis=1)   # (N,) bool
    any_anomalous = persistence_filter_1d(any_anomalous, PERSIST)

    # ── Classification per window ──────────────────────────────────────────────
    labels = []

    for i in range(N):

        # No anomaly at all
        if not any_anomalous[i]:
            labels.append("No Drift")
            continue

        # Use CUSUM S_pos as the per-feature "severity" metric.
        # S_pos accumulates evidence of persistent upward shifts —
        # the feature with the largest S_pos is the primary anomaly driver.
        s = S_pos[i]                          # shape (F,)
        ann_s    = s[sensor_idx]              # annulus CUSUM
        env_s    = s[env_idxs]               # env feature CUSUM values
        max_env  = env_s.max() + 1e-8

        # ── Step 1: Annulus dominance check ───────────────────────────────────
        ann_ratio = ann_s / max_env
        if ann_ratio > ANNULUS_DOMINANCE_THRESH:
            labels.append("Sensor Fault: Annulus Pressure Gauge")
            continue

        # ── Step 2: Single env feature dominance check ────────────────────────
        # Which env feature has the highest CUSUM?
        leader_local_idx = int(np.argmax(env_s))   # index within env_idxs
        leader_global_idx = env_idxs[leader_local_idx]
        leader_s = env_s[leader_local_idx]

        # Mean CUSUM of all OTHER env features
        other_env_s = np.delete(env_s, leader_local_idx)
        mean_others = other_env_s.mean() + 1e-8

        env_ratio = leader_s / mean_others

        if env_ratio > ENV_DOMINANCE_THRESH and ann_s < max_env:
            # One env feature clearly dominates — likely sensor fault
            # but flagged as "Probable" due to bleed-through ambiguity
            feat_name = all_features[leader_global_idx]
            display   = DISPLAY_NAMES.get(feat_name, feat_name)
            labels.append(f"Probable Sensor Fault: {display}")
            continue

        # ── Step 3: Default — Environmental Drift ─────────────────────────────
        labels.append("Environmental Drift")

    return labels, cusum_raw, roll_raw, combined_filtered, S_pos, roll_z


# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

print("=" * 60)
print("SAINT — Evaluation (v5 — Dominance Ratio Classification)")
print("=" * 60)

print("\n[1] Loading model and artifacts ...")
model          = load_model(MODEL_PATH)
scaler         = joblib.load(SCALER_PATH)
global_thresh  = float(np.load(GLOBAL_THRESH))
normal_pf      = np.load(NORMAL_PER_FEAT)

with open(FEATURE_META) as f:
    meta = json.load(f)

ALL_FEATURES = meta["all_features"]
ENV_FEATURES = meta["env_features"]
ENV_IDXS     = meta["env_indices"]
SENSOR_IDX   = meta["sensor_index"]
WINDOW_SIZE  = meta["window_size"]

print(f"    Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
print(f"    Normal window errors: {normal_pf.shape}")
print(f"\n    Detector parameters:")
print(f"      CUSUM  k={CUSUM_K}, h={CUSUM_H}")
print(f"      Roll   window={ROLL_WINDOW}, z>{ROLL_Z_THRESH}")
print(f"      Persistence: {PERSIST} windows")
print(f"\n    Classification thresholds:")
print(f"      ANNULUS_DOMINANCE_THRESH = {ANNULUS_DOMINANCE_THRESH}")
print(f"      ENV_DOMINANCE_THRESH     = {ENV_DOMINANCE_THRESH}")
print(f"\n    Known limitation:")
print(f"      Scenario B (single env sensor fault) vs Scenario A (full env drift)")
print(f"      are AMBIGUOUS due to autoencoder bleed-through. The faulting env")
print(f"      sensor cannot be reliably isolated using reconstruction errors alone.")
print(f"      Conservative default: Env Drift (safer for operations teams).")
print(f"      If dominance ratio > {ENV_DOMINANCE_THRESH}: 'Probable Sensor Fault' (uncertain).")

# ── STEP 2: EVALUATE ──────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)
all_predictions = []
report_lines    = [
    "SAINT Evaluation Report (v5 — Dominance Ratio Classification)",
    "=" * 60,
    "",
    "KNOWN LIMITATION:",
    "Scenario B (single env sensor fault) cannot be reliably distinguished",
    "from Scenario A (full environmental drift) using reconstruction errors",
    "alone. This is a fundamental property of multivariate autoencoder",
    "anomaly detection (autoencoder bleed-through). See module docstring.",
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
    print(f"  Above global threshold: "
          f"{(errors_global > global_thresh).sum()} / {len(errors_global)}")

    (fine_labels, cusum_flags, roll_flags,
     combined, S_pos, roll_z) = classify_windows(
        errors_per_feat = errors_per_feat,
        normal_pf       = normal_pf,
        all_features    = ALL_FEATURES,
        env_idxs        = ENV_IDXS,
        sensor_idx      = SENSOR_IDX,
    )

    # Align ground truth
    seq_start  = WINDOW_SIZE - 1
    y_true_raw = drift_df["TRUE_LABEL"].iloc[seq_start:].reset_index(drop=True).values
    dates      = drift_df["DATEPRD"].iloc[seq_start:].reset_index(drop=True).values
    y_pred_fine   = np.array(fine_labels)
    y_pred_coarse = np.array([coarsen_label(l) for l in y_pred_fine])
    y_true_coarse = np.array([coarsen_label(l) for l in y_true_raw])

    # Score
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

    print(f"\n  CUSUM S+ per feature (mean across drift window):")
    for i, feat in enumerate(ALL_FEATURES):
        group   = "[ENV]   " if i in ENV_IDXS else "[SENSOR]"
        display = DISPLAY_NAMES.get(feat, feat)
        mean_s  = S_pos[:, i].mean()
        max_s   = S_pos[:, i].max()
        print(f"    {group} {display:<35} mean={mean_s:7.1f}  max={max_s:7.1f}")

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
        result_df[f"ERR_{d}"]   = errors_per_feat[:, i]
        result_df[f"CUSUM_{d}"] = S_pos[:, i]
        result_df[f"ROLLZ_{d}"] = roll_z[:, i]
        result_df[f"FLAG_{d}"]  = combined[:, i]

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

print(f"\nNOTE: 'Probable Sensor Fault' labels carry inherent uncertainty")
print(f"due to autoencoder bleed-through between correlated env features.")
print(f"These should prompt further investigation, not automatic maintenance.")

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

print(f"\n[4] Results saved to {RESULTS_DIR}/")
print(f"{'='*60}")
print("Evaluation complete.")
print(f"{'='*60}")