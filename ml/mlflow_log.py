"""
Automatic MLflow logging + model registry for SAINT-OPS.

Called from train_model.py and test_model.py. Graceful fallback if server is down.

Environment:
  MLFLOW_TRACKING_URI       http://127.0.0.1:5000 (host) / http://mlflow:5000 (Docker)
  MLFLOW_EXPERIMENT_NAME    default SAINT-Drift-Detection
  MLFLOW_REGISTERED_MODEL   default SAINT
  MLFLOW_AUTO_STAGE         1 = promote new version to Staging after train (default)
  MLFLOW_AUTO_PROMOTE       1 = skip Staging and go straight to Production (legacy)
  MLFLOW_REQUIRE_APPROVAL   1 = require approved=true tag before Staging→Production
  MLFLOW_DISABLED=1         skip all logging
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

RUN_META_FILE = "mlflow_run.json"
MLFLOW_VERSION = "2.17.0"


def _enabled() -> bool:
    return os.environ.get("MLFLOW_DISABLED", "").lower() not in ("1", "true", "yes")


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "SAINT-Drift-Detection")


def registered_model_name() -> str:
    return os.environ.get("MLFLOW_REGISTERED_MODEL", "SAINT")


def _auto_stage() -> bool:
    return os.environ.get("MLFLOW_AUTO_STAGE", "1").lower() not in ("0", "false", "no")


def _auto_promote() -> bool:
    return os.environ.get("MLFLOW_AUTO_PROMOTE", "0").lower() not in ("0", "false", "no")


def _run_meta_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, RUN_META_FILE)


def save_run_meta(
    artifacts_dir: str,
    run_id: str,
    run_name: str,
    model_version: int | None = None,
    stage: str | None = None,
) -> None:
    path = _run_meta_path(artifacts_dir)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "registered_model": registered_model_name(),
    }
    if model_version is not None:
        payload["model_version"] = model_version
    if stage is not None:
        payload["stage"] = stage
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_run_meta(artifacts_dir: str) -> dict[str, Any] | None:
    path = _run_meta_path(artifacts_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_evaluation_report(path: str) -> dict[str, float]:
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

        if not mlflow.__version__.startswith("2.17."):
            print(
                f"[MLflow] Warning: client {mlflow.__version__} may not match server {MLFLOW_VERSION}. "
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


def _register_keras_model(model: Any) -> bool:
    """Register model in MLflow Model Registry (MLflow 2.9.x)."""
    import mlflow.keras

    try:
        mlflow.keras.log_model(
            model,
            artifact_path="saint-lstm-model",
            registered_model_name=registered_model_name(),
        )
        print(f"[MLflow] Registered model '{registered_model_name()}' in Model Registry")
        return True
    except Exception as exc:
        print(f"[MLflow] Registry log_model skipped ({exc}); .keras file still in Artifacts")
        return False


def promote_model_to_staging(model_name: str | None = None) -> int | None:
    """Promote the latest registered version to Staging (await comparison + approval)."""
    if not _enabled():
        return None
    try:
        from registry import promote_to_staging

        version = promote_to_staging()
        if version is not None:
            name = model_name or registered_model_name()
            print(f"[MLflow] Model '{name}' version {version} → Staging")
        return version
    except Exception as exc:
        print(f"[MLflow] Promote to Staging failed: {exc}")
        return None


def promote_model_to_production(model_name: str | None = None) -> int | None:
    """Legacy: promote latest version directly to Production (skips gate)."""
    if not _enabled():
        return None

    name = model_name or registered_model_name()

    try:
        from registry import _resolve_version_number, transition_stage

        version = _resolve_version_number(None)
        if version is None:
            print(f"[MLflow] No registered versions for '{name}' to promote")
            return None
        transition_stage(int(version), "Production", archive_existing=True)
        print(f"[MLflow] Model '{name}' version {version} → Production (direct)")
        return int(version)

    except Exception as exc:
        print(f"[MLflow] Promote to Production failed: {exc}")
        return None


def log_training_run(
    artifacts_dir: str,
    models_dir: str,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> str | None:
    if not _enabled():
        return None

    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed — pip install mlflow==2.17.0")
        return None

    _warn_if_client_version_mismatch()

    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment_name())
        run_name = f"LSTM-Autoencoder-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("stage", "training")
            _log_artifact_files(artifacts_dir, models_dir)
            registered = _register_keras_model(model)
            from mlflow.tracking import MlflowClient

            artifact_roots = [a.path for a in MlflowClient(tracking_uri()).list_artifacts(run.info.run_id)]
            if not artifact_roots:
                print(
                    "[MLflow] Warning: no artifacts on run after upload. "
                    "Ensure MLflow server uses --serve-artifacts and the train "
                    "container mounts ./mlflow-data:/mlflow."
                )
            else:
                print(f"[MLflow] Artifacts logged: {', '.join(artifact_roots)}")
                # Shared volume files are root-owned; world-read so backend can load locally if mounted.
                root = os.path.join("/mlflow", "artifacts")
                if os.path.isdir(root):
                    for dirpath, _, filenames in os.walk(root):
                        try:
                            os.chmod(dirpath, 0o755)
                        except OSError:
                            pass
                        for fn in filenames:
                            try:
                                os.chmod(os.path.join(dirpath, fn), 0o644)
                            except OSError:
                                pass

            model_version = None
            if _auto_promote():
                model_version = promote_model_to_production()
            elif _auto_stage():
                model_version = promote_model_to_staging()

            save_run_meta(
                artifacts_dir,
                run.info.run_id,
                run_name,
                model_version,
                stage="Staging" if model_version and not _auto_promote() else "Production",
            )
            print(
                f"[MLflow] Training logged — experiment '{experiment_name()}', "
                f"run '{run_name}' → {tracking_uri()}"
            )
            return run.info.run_id

    except Exception as exc:
        print(f"[MLflow] Training log failed: {exc}")
        print("         Start server: docker compose up mlflow -d")
        return None


def log_evaluation_run(
    artifacts_dir: str,
    models_dir: str,
    results_dir: str,
    metrics: dict[str, float] | None = None,
) -> None:
    if not _enabled():
        return

    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed — pip install mlflow==2.17.0")
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
                mlflow.set_tag("classifier", "fixed_val_p99_CUSUM_v8")
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
                mlflow.set_tag("classifier", "fixed_val_p99_CUSUM_v8")
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


def log_full_snapshot(artifacts_dir: str, models_dir: str, results_dir: str) -> None:
    if not _enabled():
        return

    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed — pip install mlflow==2.17.0")
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
                    "classifier": "fixed_val_p99_CUSUM_v8",
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
