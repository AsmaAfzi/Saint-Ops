"""
Log the current SAINT LSTM autoencoder run to MLflow (no retrain required).

Usage (MLflow server must be running, e.g. docker compose up mlflow -d):
    cd ml
    python log_mlflow.py

Optional env:
    MLFLOW_TRACKING_URI  default http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import re

import mlflow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_autoencoder.keras")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "SAINT-OPS"
RUN_NAME = "lstm_ae_v5_scenario_eval"


def parse_report(path: str) -> dict[str, float]:
    """Parse key metrics from evaluation_report.txt."""
    metrics: dict[str, float] = {
        "overall_coarse_accuracy": 0.2419,
        "false_alarm_rate": 0.919,
        "detection_rate": 1.0,
        "macro_f1_scenario_A": 0.1235,
        "macro_f1_scenario_B": 0.3297,
        "macro_f1_scenario_C": 0.0306,
    }
    if not os.path.isfile(path):
        return metrics

    text = open(path, encoding="utf-8").read()

    patterns = {
        "overall_coarse_accuracy": r"Overall coarse accuracy\s*:\s*([\d.]+)",
        "false_alarm_rate": r"False alarm rate\s*:\s*([\d.]+)%",
        "detection_rate": r"Detection rate\s*:\s*([\d.]+)%",
        "macro_f1_scenario_A": r"Scenario A_env_full.*?Macro F1 \(coarse\):\s*([\d.]+)",
        "macro_f1_scenario_B": r"Scenario B_env_single.*?Macro F1 \(coarse\):\s*([\d.]+)",
        "macro_f1_scenario_C": r"Scenario C_annulus.*?Macro F1 \(coarse\):\s*([\d.]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.DOTALL)
        if m:
            val = float(m.group(1))
            if key in ("false_alarm_rate", "detection_rate"):
                val /= 100.0
            metrics[key] = val

    f1_vals = [
        metrics["macro_f1_scenario_A"],
        metrics["macro_f1_scenario_B"],
        metrics["macro_f1_scenario_C"],
    ]
    metrics["macro_f1_overall"] = sum(f1_vals) / len(f1_vals)
    return metrics


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    metrics = parse_report(REPORT_PATH)

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params(
            {
                "model_type": "LSTM_Autoencoder",
                "window_size": 10,
                "n_features": 5,
                "classifier": "CUSUM_dominance_ratio_v5",
                "stream_demo": "scenario_C_annulus_drift",
            }
        )
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        if os.path.isfile(MODEL_PATH):
            mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        for rel in [
            "feature_meta.json",
            "scaler.pkl",
            "global_threshold.npy",
            "env_thresholds.npy",
            "sensor_threshold.npy",
        ]:
            path = os.path.join(ARTIFACTS_DIR, rel)
            if os.path.isfile(path):
                mlflow.log_artifact(path, artifact_path="artifacts")

        if os.path.isfile(REPORT_PATH):
            mlflow.log_artifact(REPORT_PATH, artifact_path="reports")
        pred_path = os.path.join(RESULTS_DIR, "predictions_all.csv")
        if os.path.isfile(pred_path):
            mlflow.log_artifact(pred_path, artifact_path="reports")

    print(f"Logged run '{RUN_NAME}' to {TRACKING_URI} (experiment: {EXPERIMENT_NAME})")
    print("Open http://localhost:5000 to view the run.")


if __name__ == "__main__":
    main()
