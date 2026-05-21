"""
SAINT — Evaluation Script (v5 — Dominance Ratio Classification)
Uses shared logic from inference.py.

Usage:
    python test_model.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from inference import (
    COARSE_CLASS_ORDER,
    DISPLAY_NAMES,
    coarsen_label,
    load_artifacts,
    make_sequences,
    classify_windows,
    recon_error_per_window,
    recon_error_per_feature,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SCENARIO_FILES = {
    "A_env_full": os.path.join(ROOT_DIR, "data/scenario_A_env_full.csv"),
    "B_env_single": os.path.join(ROOT_DIR, "data/scenario_B_env_single.csv"),
    "C_annulus": os.path.join(ROOT_DIR, "data/scenario_C_annulus.csv"),
}


def evaluate_scenario(name: str, path: str, bundle: dict) -> tuple[pd.DataFrame, list[str]]:
    meta = bundle["meta"]
    all_features = meta["all_features"]
    env_idxs = meta["env_indices"]
    sensor_idx = meta["sensor_index"]
    window_size = meta["window_size"]

    df_s = pd.read_csv(path, parse_dates=["DATEPRD"])
    drift_df = df_s[df_s["WINDOW"] == "drift"].copy().reset_index(drop=True)

    x_raw = drift_df[all_features].values.astype(np.float32)
    x_scaled = bundle["scaler"].transform(x_raw)
    x_seq = make_sequences(x_scaled, window_size)
    x_pred = bundle["model"].predict(x_seq, verbose=0)

    errors_global = recon_error_per_window(x_seq, x_pred)
    errors_per_feat = recon_error_per_feature(x_seq, x_pred)

    fine_labels, _, _, combined, s_pos, roll_z = classify_windows(
        errors_per_feat=errors_per_feat,
        normal_pf=bundle["normal_pf"],
        all_features=all_features,
        env_idxs=env_idxs,
        sensor_idx=sensor_idx,
    )

    seq_start = window_size - 1
    y_true_raw = drift_df["TRUE_LABEL"].iloc[seq_start:].reset_index(drop=True).values
    dates = drift_df["DATEPRD"].iloc[seq_start:].reset_index(drop=True).values
    y_pred_fine = np.array(fine_labels)
    y_pred_coarse = np.array([coarsen_label(l) for l in y_pred_fine])
    y_true_coarse = np.array([coarsen_label(l) for l in y_true_raw])

    cm = confusion_matrix(y_true_coarse, y_pred_coarse, labels=COARSE_CLASS_ORDER)
    cr = classification_report(
        y_true_coarse, y_pred_coarse, labels=COARSE_CLASS_ORDER, zero_division=0
    )
    f1_macro = f1_score(
        y_true_coarse, y_pred_coarse, labels=COARSE_CLASS_ORDER, average="macro", zero_division=0
    )

    print(f"\n{'─' * 60}")
    print(f"Scenario {name}: {path}")
    print(f"  Drift window: {len(drift_df)} rows")
    print(f"  Error range: [{errors_global.min():.4f}, {errors_global.max():.4f}]")
    print(f"  Above global threshold: {(errors_global > bundle['global_threshold']).sum()} / {len(errors_global)}")
    print(f"\n  Classification report (coarse):\n{cr}")
    print(f"  Confusion matrix {COARSE_CLASS_ORDER}\n{cm}")
    print(f"\n  Macro F1 (coarse): {f1_macro:.4f}")

    fine_counts = pd.Series(y_pred_fine).value_counts()
    print("\n  Fine-grained breakdown:")
    for label, cnt in fine_counts.items():
        print(f"    {label:<55} {cnt:4d} ({cnt / len(y_pred_fine) * 100:.1f}%)")

    result_df = pd.DataFrame({
        "SCENARIO": name,
        "DATEPRD": dates,
        "TRUE_LABEL": y_true_raw,
        "PREDICTED_LABEL_FINE": y_pred_fine,
        "PREDICTED_LABEL_COARSE": y_pred_coarse,
        "GLOBAL_ERROR": errors_global,
        "CORRECT_COARSE": y_true_coarse == y_pred_coarse,
    })
    for i, feat in enumerate(all_features):
        d = DISPLAY_NAMES.get(feat, feat).replace(" ", "_")
        result_df[f"ERR_{d}"] = errors_per_feat[:, i]
        result_df[f"CUSUM_{d}"] = s_pos[:, i]
        result_df[f"ROLLZ_{d}"] = roll_z[:, i]
        result_df[f"FLAG_{d}"] = combined[:, i]

    report_block = [
        f"\n{'=' * 60}",
        f"Scenario {name}: {path}",
        f"{'=' * 60}",
        cr,
        f"Confusion matrix {COARSE_CLASS_ORDER}",
        str(cm),
        f"\nMacro F1 (coarse): {f1_macro:.4f}",
        "\nFine-grained breakdown:",
        fine_counts.to_string(),
    ]
    return result_df, report_block


def main() -> None:
    print("=" * 60)
    print("SAINT — Evaluation (v5 — Dominance Ratio Classification)")
    print("=" * 60)

    print("\n[1] Loading model and artifacts ...")
    bundle = load_artifacts(ARTIFACTS_DIR, MODELS_DIR)
    meta = bundle["meta"]
    print(f"    Features ({len(meta['all_features'])}): {meta['all_features']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_predictions = []
    report_lines = [
        "SAINT Evaluation Report (v5 — Dominance Ratio Classification)",
        "=" * 60,
    ]

    for name, path in SCENARIO_FILES.items():
        result_df, block = evaluate_scenario(name, path, bundle)
        all_predictions.append(result_df)
        report_lines.extend(block)

    all_df = pd.concat(all_predictions, ignore_index=True)
    overall_acc = all_df["CORRECT_COARSE"].mean()

    no_d = all_df[all_df["TRUE_LABEL"].apply(coarsen_label) == "No Drift"]
    fa = (no_d["PREDICTED_LABEL_COARSE"] != "No Drift").sum()
    fa_r = fa / len(no_d) if len(no_d) > 0 else 0.0

    dr = all_df[all_df["TRUE_LABEL"].apply(coarsen_label) != "No Drift"]
    det = (dr["PREDICTED_LABEL_COARSE"] != "No Drift").sum()
    det_r = det / len(dr) if len(dr) > 0 else 0.0

    print(f"\n{'=' * 60}\nOVERALL SUMMARY\n{'=' * 60}")
    print(f"Overall coarse accuracy: {overall_acc:.4f}")
    print(f"False alarm rate: {fa_r * 100:.1f}%  ({fa}/{len(no_d)})")
    print(f"Detection rate: {det_r * 100:.1f}%  ({det}/{len(dr)})")

    all_df.to_csv(os.path.join(RESULTS_DIR, "predictions_all.csv"), index=False)
    report_lines += [
        f"\n{'=' * 60}",
        "OVERALL",
        f"{'=' * 60}",
        f"Overall coarse accuracy : {overall_acc:.4f}",
        f"False alarm rate        : {fa_r * 100:.1f}%",
        f"Detection rate          : {det_r * 100:.1f}%",
    ]
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[4] Results saved to {RESULTS_DIR}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
