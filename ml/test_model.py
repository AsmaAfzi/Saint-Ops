"""
SAINT — Evaluation Script (v8)
================================
Input:
  models/lstm_autoencoder.keras
  artifacts/scaler.pkl
  artifacts/global_threshold.npy
  artifacts/normal_errors_per_feature.npy   ← validation window errors
  artifacts/feature_meta.json
  scenario_A_env_full.csv
  scenario_B_env_single.csv
  scenario_C_annulus.csv

Outputs:
  results/predictions_all.csv
  results/evaluation_report.txt

────────────────────────────────────────────────────────────
CHANGES FROM v7 — TWO TARGETED FIXES
────────────────────────────────────────────────────────────
Fix 1: Fixed val_p99 normalisation replaces rolling p99.

  v7 used rolling_p99_normalize() — each feature's error was divided by
  a rolling p99 computed over the previous 50 windows of the TEST data.
  Because the injected drift is a gradual ramp, the rolling p99 adapted
  to the growing signal at nearly the same rate. The result: all features
  ended up with norm_error ≈ 0.7-0.9 regardless of whether they were
  injected. The denominator was chasing the numerator.

  v8 uses fixed_p99_normalize() — each feature's error is divided by
  the p99 of that feature's reconstruction errors in the VALIDATION
  WINDOW (Dec 2009–Apr 2010), loaded from normal_errors_per_feature.npy.
  This gives a stable, pre-injection reference:
    norm_error = 1.0 → feature is at its validation p99 boundary
    norm_error > 1.0 → feature error is above its normal ceiling
    norm_error < 1.0 → feature error is below its normal ceiling

  Non-injected features stay near norm_error ≈ 0.7-0.9 (same as before,
  they are within their normal range). Injected features grow above 1.0
  as drift accumulates. The difference is now fixed and detectable.

Fix 2: CUSUM allowance k raised from 0.5 to 0.9.

  The CUSUM allowance k defines what the detector "expects" as the
  normal level. Every step, k is subtracted from the normalised error
  before accumulation: S[i] = max(0, S[i-1] + norm_error[i] - k).

  With k=0.5 and norm_error ≈ 0.8 (non-injected, within normal):
    S[i] = max(0, S[i-1] + 0.8 - 0.5) = S[i-1] + 0.3
    After 164 steps: S ≈ 49 — CUSUM grows continuously on CLEAN data.
    All features accumulate large CUSUM regardless of injection.

  With k=0.9 and norm_error ≈ 0.8 (non-injected, within normal):
    S[i] = max(0, S[i-1] + 0.8 - 0.9) = max(0, S[i-1] - 0.1) → 0
    CUSUM stays near zero on clean features. ✓

  With k=0.9 and norm_error > 1.0 (injected, growing above val_p99):
    S[i] = max(0, S[i-1] + 1.1 - 0.9) = S[i-1] + 0.2 → grows.
    CUSUM accumulates only on features that consistently exceed val_p99. ✓

  k=0.9 is set at the midpoint between expected normal (0.8) and
  expected anomalous (1.0+). This is the standard SPC choice:
  k = (mu_normal + mu_anomalous) / 2 = (0.8 + 1.0) / 2 = 0.9.

  h=4.0 unchanged — alarm fires after 4 units accumulated above k.
  With k=0.9 and clean data (norm_error=0.8): CUSUM trends to 0,
  no alarm. With injected drift (norm_error>1.0 growing): alarm fires
  within ~20-40 steps of the drift exceeding the val_p99 ceiling.

Expected outcome after these two fixes:
  Non-injected features: CUSUM ≈ 0-2 (stays near zero)
  Injected features:     CUSUM ≈ 15-50 (grows persistently)
  Dominance ratios:      5-30x (clearly exceeds thresholds of 2.0/2.5)
  Scenario A: env features all dominate → Environmental Drift ✓
  Scenario B: DH_PRESS dominates (slightly) → Probable Sensor Fault ✓
  Scenario C: ANNULUS dominates clearly → Sensor Fault: Annulus ✓

────────────────────────────────────────────────────────────
KNOWN LIMITATION (unchanged)
────────────────────────────────────────────────────────────
Scenario B (single env sensor fault) vs full env drift: ambiguous
due to autoencoder bleed-through. DH_PRESS fault causes model to
expect correlated features (WHP, DH_TEMP) to follow — they don't —
so their reconstruction error also rises. DH_PRESS will have slightly
higher CUSUM but the gap may not always exceed ENV_DOMINANCE_THRESH.
Conservative default: Environmental Drift when ambiguous.

Usage:
    python evaluate_saint.py
    (no retraining needed — only evaluate_saint.py changed)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from inference import (
    ANNULUS_DOMINANCE_THRESH,
    COARSE_CLASS_ORDER,
    CUSUM_H,
    CUSUM_K,
    DISPLAY_NAMES,
    ENV_DOMINANCE_THRESH,
    MAX_NORMALIZED_ERROR,
    PERSIST,
    ROLL_Z_THRESH,
    ROLL_Z_WINDOW,
    classify_windows,
    coarsen_label,
    compute_val_p99,
    make_sequences,
    recon_error_per_feature,
    recon_error_per_window,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SCENARIO_FILES = {
    "A_env_full":   os.path.join(ROOT_DIR, "data", "scenario_A_env_full.csv"),
    "B_env_single": os.path.join(ROOT_DIR, "data", "scenario_B_env_single.csv"),
    "C_annulus":    os.path.join(ROOT_DIR, "data", "scenario_C_annulus.csv"),
}

MODEL_PATH    = os.path.join(ROOT_DIR, "models",    "lstm_autoencoder.keras")
SCALER_PATH   = os.path.join(ROOT_DIR, "artifacts", "scaler.pkl")
GLOBAL_THRESH = os.path.join(ROOT_DIR, "artifacts", "global_threshold.npy")
NORMAL_PF     = os.path.join(ROOT_DIR, "artifacts", "normal_errors_per_feature.npy")
FEATURE_META  = os.path.join(ROOT_DIR, "artifacts", "feature_meta.json")

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

print("=" * 60)
print("SAINT — Evaluation (v8 — Fixed Val_P99 Normalisation, k=0.9)")
print("=" * 60)

print("\n[1] Loading model and artifacts ...")
model         = load_model(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
global_thresh = float(np.load(GLOBAL_THRESH))
normal_pf     = np.load(NORMAL_PF)   # shape (94, 5) — validation window errors

with open(FEATURE_META) as f:
    meta = json.load(f)

ALL_FEATURES = meta["all_features"]
ENV_FEATURES = meta["env_features"]
ENV_IDXS     = meta["env_indices"]
SENSOR_IDX   = meta["sensor_index"]
WINDOW_SIZE  = meta["window_size"]

val_p99 = compute_val_p99(normal_pf)

print(f"    Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
print(f"    Validation window errors: {normal_pf.shape}")
print(f"    Fixed val_p99 per feature:")
for i, feat in enumerate(ALL_FEATURES):
    display = DISPLAY_NAMES.get(feat, feat)
    print(f"      {display:<35} p99={val_p99[i]:.6f}")
print(f"    Global threshold: {global_thresh:.6f}")
print(f"\n    Detection parameters:")
print(f"      CUSUM              k={CUSUM_K} (was 0.5), h={CUSUM_H}")
print(f"      Normalisation      fixed val_p99 (was rolling p99)")
print(f"      Error cap          {MAX_NORMALIZED_ERROR}x val_p99")
print(f"      Rolling z          window={ROLL_Z_WINDOW}, thresh={ROLL_Z_THRESH}")
print(f"      Persistence        {PERSIST} windows")
print(f"      Annulus dominance  > {ANNULUS_DOMINANCE_THRESH}x")
print(f"      Env dominance      > {ENV_DOMINANCE_THRESH}x")

# ── STEP 2: EVALUATE ──────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)
all_predictions = []
report_lines    = [
    "SAINT Evaluation Report (v8 — Fixed Val_P99 Normalisation)",
    "=" * 60,
    "",
    "KNOWN LIMITATION:",
    "Scenario B (single env sensor fault) vs full env drift: ambiguous",
    "due to autoencoder bleed-through. Conservative default: Env Drift.",
    "Windows where one env feature dominates CUSUM by > 2.5x labelled",
    "'Probable Sensor Fault: X' with inherent uncertainty.",
    "",
]

for scenario_name, scenario_file in SCENARIO_FILES.items():
    print(f"\n{'─'*60}")
    print(f"Scenario {scenario_name}: {scenario_file}")

    df_s     = pd.read_csv(scenario_file, parse_dates=["DATEPRD"])
    drift_df = df_s[df_s["WINDOW"] == "drift"].copy().reset_index(drop=True)
    print(f"  Drift window: {len(drift_df)} rows  "
          f"({drift_df['DATEPRD'].min().date()} → "
          f"{drift_df['DATEPRD'].max().date()})")

    X_raw    = drift_df[ALL_FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    X_seq    = make_sequences(X_scaled, WINDOW_SIZE)
    X_pred   = model.predict(X_seq, verbose=0)

    errors_global   = recon_error_per_window(X_seq, X_pred)
    errors_per_feat = recon_error_per_feature(X_seq, X_pred)

    print(f"  Error range:      [{errors_global.min():.4f}, {errors_global.max():.4f}]")
    print(f"  Global threshold:  {global_thresh:.4f}")
    print(f"  Above threshold:   "
          f"{(errors_global > global_thresh).sum()} / {len(errors_global)}")

    (fine_labels, cusum_flags, roll_flags,
     combined, S_pos, roll_z, norm_errors) = classify_windows(
        errors_per_feat = errors_per_feat,
        val_p99         = val_p99,
        all_features    = ALL_FEATURES,
        env_idxs        = ENV_IDXS,
        sensor_idx      = SENSOR_IDX,
    )

    seq_start     = WINDOW_SIZE - 1
    y_true_raw    = (drift_df["TRUE_LABEL"]
                     .iloc[seq_start:].reset_index(drop=True).values)
    dates         = (drift_df["DATEPRD"]
                     .iloc[seq_start:].reset_index(drop=True).values)
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

    print(f"\n  CUSUM S+ per feature (mean | max)  [k={CUSUM_K}, h={CUSUM_H}]:")
    for i, feat in enumerate(ALL_FEATURES):
        group   = "[ENV]   " if i in ENV_IDXS else "[SENSOR]"
        display = DISPLAY_NAMES.get(feat, feat)
        print(f"    {group} {display:<35} "
              f"mean={S_pos[:,i].mean():7.2f}  max={S_pos[:,i].max():7.2f}")

    print(f"\n  Fixed-normalised errors (mean | max)  [ref=val_p99]:")
    for i, feat in enumerate(ALL_FEATURES):
        display = DISPLAY_NAMES.get(feat, feat)
        print(f"    {display:<35} "
              f"mean={norm_errors[:,i].mean():.3f}  max={norm_errors[:,i].max():.3f}  "
              f"val_p99={val_p99[i]:.4f}")

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
print(f"Prompts engineer investigation, not automatic maintenance action.")

# ── STEP 3b: HEADLINE METRICS (SAINT / FYP reporting protocol) ───────────────
# Primary metrics for the project report (consistent with the SAINT mini-project
# interim evaluation). These differ from coarse 3-class macro-F1, which is
# reported separately as extended disambiguation analysis.

def _binary_drift_metrics(true_labels, pred_coarse):
    yt = np.array([coarsen_label(l) != "No Drift" for l in true_labels])
    yp = np.array([p != "No Drift" for p in pred_coarse])
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(yt) if len(yt) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_alarm_rate": far,
        "detection_rate": rec,
    }


def _class_f1(true_labels, pred_coarse, target_class):
    yt = np.array([coarsen_label(l) == target_class for l in true_labels])
    yp = np.array([p == target_class for p in pred_coarse])
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


binary_all = _binary_drift_metrics(
    all_df["TRUE_LABEL"], all_df["PREDICTED_LABEL_COARSE"]
)
s_b = all_df[all_df["SCENARIO"] == "B_env_single"]
s_a = all_df[all_df["SCENARIO"] == "A_env_full"]
sensor_path = _class_f1(
    s_b["TRUE_LABEL"], s_b["PREDICTED_LABEL_COARSE"], "Sensor Drift"
)
env_path = _class_f1(
    s_a["TRUE_LABEL"], s_a["PREDICTED_LABEL_COARSE"], "Environmental Drift"
)

# Validation-baseline FAR: baseline + normal windows (no injected drift labels)
baseline_fa = 0
baseline_n = 0
for scenario_name, scenario_file in SCENARIO_FILES.items():
    df_clean = pd.read_csv(scenario_file, parse_dates=["DATEPRD"])
    clean = df_clean[df_clean["WINDOW"].isin(["baseline", "normal"])].copy()
    if len(clean) < WINDOW_SIZE:
        continue
    X_raw = clean[ALL_FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    X_seq = make_sequences(X_scaled, WINDOW_SIZE)
    X_pred = model.predict(X_seq, verbose=0)
    errors_per_feat = recon_error_per_feature(X_seq, X_pred)
    fine_labels, *_ = classify_windows(
        errors_per_feat=errors_per_feat,
        val_p99=val_p99,
        all_features=ALL_FEATURES,
        env_idxs=ENV_IDXS,
        sensor_idx=SENSOR_IDX,
    )
    pred_coarse = np.array([coarsen_label(l) for l in fine_labels])
    baseline_n += len(pred_coarse)
    baseline_fa += int(np.sum(pred_coarse != "No Drift"))

baseline_far = baseline_fa / baseline_n if baseline_n else 0.0

headline_global_f1 = float(
    np.mean([binary_all["f1"], sensor_path["f1"], env_path["f1"]])
)
headline_primary_f1 = float(np.mean([binary_all["f1"], sensor_path["f1"]]))

print(f"\n{'='*60}")
print("HEADLINE METRICS (SAINT / FYP reporting protocol)")
print(f"{'='*60}")
print(
    f"  Binary drift-presence F1     : {binary_all['f1']:.4f}  "
    f"(acc={binary_all['accuracy']:.4f}, prec={binary_all['precision']:.4f}, "
    f"rec={binary_all['recall']:.4f})"
)
print(
    f"  Sensor-pathway F1 (Scen. B) : {sensor_path['f1']:.4f}  "
    f"(prec={sensor_path['precision']:.4f}, rec={sensor_path['recall']:.4f})"
)
print(
    f"  Environmental-pathway F1 (A): {env_path['f1']:.4f}  "
    f"(prec={env_path['precision']:.4f}, rec={env_path['recall']:.4f})"
)
print(f"  Headline primary F1 (B+S)   : {headline_primary_f1:.4f}")
print(f"  Headline global F1 (mean)   : {headline_global_f1:.4f}")
print(
    f"  Validation-baseline FAR     : {baseline_far*100:.1f}%  "
    f"({baseline_fa}/{baseline_n} windows on baseline+normal data)"
)
print(
    f"  Injected-window FAR         : {fa_r*100:.1f}%  "
    f"(pre-ramp no-drift rows inside drift evaluation window)"
)

# ── STEP 4: SAVE ──────────────────────────────────────────────────────────────

all_df.to_csv(os.path.join(RESULTS_DIR, "predictions_all.csv"), index=False)

report_lines += [
    f"\n{'='*60}", "OVERALL",
    f"{'='*60}",
    f"Overall coarse accuracy : {overall_acc:.4f}",
    f"False alarm rate        : {fa_r*100:.1f}%",
    f"Detection rate          : {det_r*100:.1f}%",
    "\nFine-grained distribution:", all_fine.to_string(),
    "\nNOTE: 'Probable Sensor Fault' labels carry inherent uncertainty.",
    "Autoencoder bleed-through between correlated env features means",
    "single-feature env faults cannot be isolated with certainty.",
    f"\n{'='*60}",
    "HEADLINE METRICS (SAINT / FYP reporting protocol)",
    f"{'='*60}",
    "Primary metrics for project reporting (mini-project protocol).",
    "Coarse 3-class macro-F1 above is supplementary disambiguation analysis.",
    f"Binary drift-presence F1     : {binary_all['f1']:.4f}",
    f"Binary drift accuracy        : {binary_all['accuracy']:.4f}",
    f"Binary drift precision       : {binary_all['precision']:.4f}",
    f"Binary drift recall          : {binary_all['recall']:.4f}",
    f"Sensor-pathway F1 (Scen. B)  : {sensor_path['f1']:.4f}",
    f"Environmental-pathway F1 (A) : {env_path['f1']:.4f}",
    f"Headline primary F1 (B+S)    : {headline_primary_f1:.4f}",
    f"Headline global F1 (mean)    : {headline_global_f1:.4f}",
    f"Validation-baseline FAR      : {baseline_far*100:.1f}%",
    f"Injected-window FAR          : {fa_r*100:.1f}%",
]
with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print(f"\n[4] Results saved to '{RESULTS_DIR}/'")

scenario_f1 = {}
for sn in SCENARIO_FILES:
    s = all_df[all_df["SCENARIO"] == sn]
    scenario_f1[sn] = f1_score(
        [coarsen_label(l) for l in s["TRUE_LABEL"]],
        s["PREDICTED_LABEL_COARSE"],
        labels=COARSE_CLASS_ORDER,
        average="macro",
        zero_division=0,
    )
macro_f1_overall = sum(scenario_f1.values()) / len(scenario_f1)

# ── STEP 5: MLFLOW ────────────────────────────────────────────────────────────
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
            "macro_f1_scenario_A": float(scenario_f1.get("A_env_full", 0.0)),
            "macro_f1_scenario_B": float(scenario_f1.get("B_env_single", 0.0)),
            "macro_f1_scenario_C": float(scenario_f1.get("C_annulus", 0.0)),
            "macro_f1_overall": float(macro_f1_overall),
            "headline_primary_f1": headline_primary_f1,
            "headline_binary_f1": float(binary_all["f1"]),
            "headline_sensor_pathway_f1": float(sensor_path["f1"]),
            "headline_env_pathway_f1": float(env_path["f1"]),
            "headline_global_f1": float(headline_global_f1),
            "validation_baseline_far": float(baseline_far),
        },
    )
except Exception as _mlflow_exc:
    print(f"[MLflow] {_mlflow_exc}")

print(f"{'='*60}")
print("Evaluation complete.")
print(f"{'='*60}")