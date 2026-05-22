"""
Automatic MLflow logging for SAINT-OPS training and evaluation.

Uses log_artifact only (compatible with MLflow 2.9.x server + client).
Avoids mlflow.keras.log_model / logged-models API (404 on server 2.9.2).

Environment:
  MLFLOW_TRACKING_URI    default http://127.0.0.1:5000
  MLFLOW_EXPERIMENT_NAME default SAINT-OPS
  MLFLOW_DISABLED=1      skip all logging

Install matching client: pip install mlflow==2.9.2  (see ml/requirements.txt)
Start server: docker compose up mlflow -d
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

RUN_META_FILE = "mlflow_run.json"
MLFLOW_VERSION = "2.9.2"


def _enabled() -> bool:
    return os.environ.get("MLFLOW_DISABLED", "").lower() not in ("1", "true", "yes")


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "SAINT-OPS")


def _run_meta_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, RUN_META_FILE)


def save_run_meta(artifacts_dir: str, run_id: str, run_name: str) -> None:
    path = _run_meta_path(artifacts_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "run_name": run_name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


def load_run_meta(artifacts_dir: str) -> dict[str, str] | None:
    path = _run_meta_path(artifacts_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_evaluation_report(path: str) -> dict[str, float]:
    """Parse metrics from evaluation_report.txt."""
    metrics: dict[str, float] = {}
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

    if metrics:
        f1_vals = [
            metrics.get("macro_f1_scenario_A", 0.0),
            metrics.get("macro_f1_scenario_B", 0.0),
            metrics.get("macro_f1_scenario_C", 0.0),
        ]
        metrics["macro_f1_overall"] = sum(f1_vals) / len(f1_vals)
    return metrics


def _warn_if_client_version_mismatch() -> None:
    try:
        import mlflow

        client_ver = mlflow.__version__
        if not client_ver.startswith("2.9."):
            print(
                f"[MLflow] Warning: client {client_ver} may not match server {MLFLOW_VERSION}. "
                f"Run: pip install mlflow=={MLFLOW_VERSION}"
            )
    except ImportError:
        pass


def _log_artifact_files(artifacts_dir: str, models_dir: str) -> None:
    import mlflow

    model_path = os.path.join(models_dir, "lstm_autoencoder.keras")
    if os.path.isfile(model_path):
        mlflow.log_artifact(model_path, artifact_path="model")

    for rel in [
        "feature_meta.json",
        "scaler.pkl",
        "global_threshold.npy",
        "env_thresholds.npy",
        "sensor_threshold.npy",
        "normal_errors_per_feature.npy",
        "normal_errors.npy",
    ]:
        path = os.path.join(artifacts_dir, rel)
        if os.path.isfile(path):
            mlflow.log_artifact(path, artifact_path="artifacts")


def log_training_run(
    artifacts_dir: str,
    models_dir: str,
    model: Any,  # kept for API compatibility; model logged via .keras file on disk
    params: dict[str, Any],
    metrics: dict[str, float],
) -> str | None:
    """
    Start a new MLflow run after training. Saves run_id for evaluation to extend.
    """
    del model  # logged as artifact from models_dir, not via log_model API

    if not _enabled():
        return None

    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed — pip install mlflow==2.9.2")
        return None

    _warn_if_client_version_mismatch()

    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment_name())
        run_name = f"saint_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("stage", "training")
            mlflow.set_tag("model_file", "lstm_autoencoder.keras")
            _log_artifact_files(artifacts_dir, models_dir)

            save_run_meta(artifacts_dir, run.info.run_id, run_name)
            print(
                f"[MLflow] Training logged — run '{run_name}' "
                f"(id {run.info.run_id[:8]}…) → {tracking_uri()}"
            )
            print(f"         UI: {tracking_uri()}/#/experiments")
            return run.info.run_id

    except Exception as exc:
        print(f"[MLflow] Training log failed: {exc}")
        print(f"         Ensure server is up: docker compose up mlflow -d")
        print(f"         Client version: pip install mlflow=={MLFLOW_VERSION}")
        return None


def log_evaluation_run(
    artifacts_dir: str,
    models_dir: str,
    results_dir: str,
    metrics: dict[str, float] | None = None,
) -> None:
    """
    Append evaluation metrics/artifacts to the latest training run, or open a new run.
    """
    if not _enabled():
        return

    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed — pip install mlflow==2.9.2")
        return

    _warn_if_client_version_mismatch()

    report_path = os.path.join(results_dir, "evaluation_report.txt")
    pred_path = os.path.join(results_dir, "predictions_all.csv")

    if metrics is None:
        metrics = parse_evaluation_report(report_path)

    meta = load_run_meta(artifacts_dir)
    run_id = meta.get("run_id") if meta else None

    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment_name())

        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("stage", "evaluation")
                if metrics:
                    mlflow.log_metrics(metrics)
                if os.path.isfile(report_path):
                    mlflow.log_artifact(report_path, artifact_path="reports")
                if os.path.isfile(pred_path):
                    mlflow.log_artifact(pred_path, artifact_path="reports")
                _log_artifact_files(artifacts_dir, models_dir)
            target = run_id
        else:
            run_name = f"saint_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("stage", "evaluation")
                if metrics:
                    mlflow.log_metrics(metrics)
                if os.path.isfile(report_path):
                    mlflow.log_artifact(report_path, artifact_path="reports")
                if os.path.isfile(pred_path):
                    mlflow.log_artifact(pred_path, artifact_path="reports")
                _log_artifact_files(artifacts_dir, models_dir)
            target = run_name

        print(f"[MLflow] Evaluation logged to run {target} → {tracking_uri()}")

    except Exception as exc:
        print(f"[MLflow] Evaluation log failed: {exc}")
        print(f"         Ensure server is up: docker compose up mlflow -d")


def log_full_snapshot(artifacts_dir: str, models_dir: str, results_dir: str) -> None:
    """Manual: log current files + report metrics in one run."""
    if not _enabled():
        return

    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed — pip install mlflow==2.9.2")
        return

    _warn_if_client_version_mismatch()

    report_path = os.path.join(results_dir, "evaluation_report.txt")
    metrics = parse_evaluation_report(report_path)

    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment_name())
        run_name = f"saint_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model_type": "LSTM_Autoencoder",
                    "training": "v2_non_drift_val_thresholds",
                    "classifier": "rolling_p99_CUSUM_v7",
                }
            )
            if metrics:
                mlflow.log_metrics(metrics)
            _log_artifact_files(artifacts_dir, models_dir)
            if os.path.isfile(report_path):
                mlflow.log_artifact(report_path, artifact_path="reports")
            pred_path = os.path.join(results_dir, "predictions_all.csv")
            if os.path.isfile(pred_path):
                mlflow.log_artifact(pred_path, artifact_path="reports")

        print(f"[MLflow] Snapshot logged as '{run_name}' → {tracking_uri()}")

    except Exception as exc:
        print(f"[MLflow] Snapshot failed: {exc}")
