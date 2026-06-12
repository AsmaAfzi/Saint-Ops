"""
Shared SAINT inference: LSTM reconstruction + fixed val_p99 CUSUM + dominance-ratio classification (v8).
Used by test_model.py (evaluation) and backend.py (stream precompute).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ── Detector / classifier parameters (v8) ─────────────────────────────────────

CUSUM_K = 0.9
CUSUM_H = 4.0
MAX_NORMALIZED_ERROR = 5.0
ROLL_Z_WINDOW = 7
ROLL_Z_THRESH = 2.0
PERSIST = 3
ANNULUS_DOMINANCE_THRESH = 2.0
ENV_DOMINANCE_THRESH = 2.5

DISPLAY_NAMES = {
    "AVG_DOWNHOLE_PRESSURE": "Downhole Pressure Gauge",
    "AVG_DOWNHOLE_TEMPERATURE": "Downhole Temperature Sensor",
    "BORE_OIL_VOL": "Oil Flow Meter",
    "AVG_WHP_P": "Wellhead Pressure Gauge",
    "AVG_ANNULUS_PRESS": "Annulus Pressure Gauge",
}

COARSE_CLASS_ORDER = ["No Drift", "Environmental Drift", "Sensor Drift"]


def coarsen_label(label: str) -> str:
    if label == "No Drift":
        return "No Drift"
    if label in ("Environmental Drift", "Probable Environmental Drift"):
        return "Environmental Drift"
    return "Sensor Drift"


def make_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.array([data[i : i + window_size] for i in range(len(data) - window_size + 1)])


def recon_error_per_window(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.square(x_true - x_pred), axis=(1, 2))


def recon_error_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.square(x_true - x_pred), axis=1)


def compute_val_p99(normal_pf: np.ndarray) -> np.ndarray:
    """Fixed p99 per feature from validation-window reconstruction errors."""
    return np.percentile(normal_pf, 99, axis=0)


def fixed_p99_normalize(
    errors_per_feat: np.ndarray, val_p99: np.ndarray, cap: float
) -> np.ndarray:
    """Normalise each feature's error by its fixed validation-window p99."""
    val_p99_safe = np.where(val_p99 < 1e-8, 1e-8, val_p99)
    normalized = errors_per_feat / val_p99_safe[np.newaxis, :]
    return np.clip(normalized, 0, cap)


def compute_cusum(
    normalized_errors: np.ndarray, k: float, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, _ = normalized_errors.shape
    s_pos = np.zeros((n, normalized_errors.shape[1]))
    s_neg = np.zeros((n, normalized_errors.shape[1]))
    for i in range(1, n):
        s_pos[i] = np.maximum(0, s_pos[i - 1] + normalized_errors[i] - k)
        s_neg[i] = np.maximum(0, s_neg[i - 1] - normalized_errors[i] - k)
    flags = (s_pos > h) | (s_neg > h)
    return flags, s_pos, s_neg


def compute_rolling_z(
    errors_per_feat: np.ndarray, window: int, z_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(errors_per_feat)
    roll_mean = df.rolling(window, min_periods=3).mean()
    roll_std = df.rolling(window, min_periods=3).std().replace(0, 1e-8).fillna(1e-8)
    z = np.nan_to_num((df - roll_mean).div(roll_std).values, nan=0.0)
    return z > z_thresh, z


def persistence_filter_2d(flags: np.ndarray, persist: int) -> np.ndarray:
    n, f = flags.shape
    filtered = np.zeros_like(flags, dtype=bool)
    for feat in range(f):
        for i in range(persist - 1, n):
            if flags[i - persist + 1 : i + 1, feat].all():
                filtered[i, feat] = True
    return filtered


def persistence_filter_1d(flags: np.ndarray, persist: int) -> np.ndarray:
    filtered = np.zeros_like(flags, dtype=bool)
    for i in range(persist - 1, len(flags)):
        if flags[i - persist + 1 : i + 1].all():
            filtered[i] = True
    return filtered


def classify_windows(
    errors_per_feat: np.ndarray,
    val_p99: np.ndarray,
    all_features: list[str],
    env_idxs: list[int],
    sensor_idx: int,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    norm_errors = fixed_p99_normalize(errors_per_feat, val_p99, MAX_NORMALIZED_ERROR)
    cusum_raw, s_pos, _ = compute_cusum(norm_errors, CUSUM_K, CUSUM_H)
    roll_raw, roll_z = compute_rolling_z(errors_per_feat, ROLL_Z_WINDOW, ROLL_Z_THRESH)

    combined_raw = cusum_raw | roll_raw
    combined_filtered = persistence_filter_2d(combined_raw, PERSIST)
    any_anomalous = persistence_filter_1d(combined_filtered.any(axis=1), PERSIST)

    labels: list[str] = []
    for i in range(errors_per_feat.shape[0]):
        if not any_anomalous[i]:
            labels.append("No Drift")
            continue

        s = s_pos[i]
        ann_s = s[sensor_idx]
        env_s = s[env_idxs]
        max_env = env_s.max() + 1e-8

        if ann_s / max_env > ANNULUS_DOMINANCE_THRESH:
            labels.append("Sensor Fault: Annulus Pressure Gauge")
            continue

        leader_local = int(np.argmax(env_s))
        leader_s = env_s[leader_local]
        others_mean = np.delete(env_s, leader_local).mean() + 1e-8

        if leader_s / others_mean > ENV_DOMINANCE_THRESH and ann_s < max_env:
            feat_name = all_features[env_idxs[leader_local]]
            display = DISPLAY_NAMES.get(feat_name, feat_name)
            labels.append(f"Probable Sensor Fault: {display}")
            continue

        labels.append("Environmental Drift")

    return (
        labels,
        cusum_raw,
        roll_raw,
        combined_filtered,
        s_pos,
        roll_z,
        norm_errors,
    )


def _default_registry_uri(stage: str = "Production") -> str:
    name = os.environ.get("MLFLOW_REGISTERED_MODEL", "SAINT")
    return os.environ.get("MLFLOW_MODEL_URI", f"models:/{name}/{stage}")


def resolve_registry_run_id(registry_uri: str) -> str | None:
    """Resolve models:/NAME/STAGE to the backing MLflow run_id."""
    if not registry_uri.startswith("models:/"):
        return None
    parts = registry_uri.replace("models:/", "").split("/")
    if len(parts) < 2:
        return None
    name, stage = parts[0], parts[1]
    try:
        from mlflow.tracking import MlflowClient

        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        client = MlflowClient(tracking_uri=uri)
        versions = client.get_latest_versions(name, stages=[stage])
        if not versions:
            return None
        return versions[0].run_id
    except Exception:
        return None


def _load_keras_model(
    models_dir: str,
    registry_uri: str | None = None,
) -> tuple[Any, str, str | None]:
    """
    Load Keras model from MLflow Model Registry with local fallback.
    Returns (model, source, run_id).
    """
    registry_uri = registry_uri or _default_registry_uri("Production")
    local_path = os.path.join(models_dir, "lstm_autoencoder.keras")
    run_id: str | None = None

    if os.environ.get("MLFLOW_DISABLED", "").lower() in ("1", "true", "yes"):
        return load_model(local_path), "local", None

    try:
        import mlflow
        import mlflow.keras

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        run_id = resolve_registry_run_id(registry_uri)
        cache_dir = tempfile.mkdtemp(prefix="saint_mlflow_")
        model = mlflow.keras.load_model(registry_uri, dst_path=cache_dir)
        print(f"[MLflow] Model loaded from registry: {registry_uri}")
        return model, "registry", run_id
    except Exception as exc:
        print(f"[MLflow] Registry unavailable ({type(exc).__name__}: {exc}), loading local file")
        return load_model(local_path), "local", None


def _artifact_base_dir(artifacts_dir: str, run_id: str | None) -> str:
    """Prefer MLflow run artifacts when model came from registry."""
    if not run_id:
        return artifacts_dir
    try:
        from mlflow.tracking import MlflowClient

        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        cache = tempfile.mkdtemp(prefix="saint_mlflow_art_")
        client = MlflowClient(tracking_uri=uri)
        downloaded = client.download_artifacts(run_id, "artifacts", cache)
        base = downloaded if os.path.isdir(downloaded) else cache
        if os.path.isfile(os.path.join(base, "feature_meta.json")):
            print(f"[MLflow] Threshold artifacts loaded from run {run_id}")
            return base
    except Exception as exc:
        print(f"[MLflow] Run artifact download failed ({exc}), using local artifacts/")
    return artifacts_dir


def load_artifacts(
    artifacts_dir: str,
    models_dir: str,
    registry_uri: str | None = None,
) -> dict[str, Any]:
    model, model_source, run_id = _load_keras_model(models_dir, registry_uri=registry_uri)
    art_dir = _artifact_base_dir(artifacts_dir, run_id if model_source == "registry" else None)

    meta_path = os.path.join(art_dir, "feature_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "model": model,
        "model_source": model_source,
        "registry_run_id": run_id,
        "registry_uri": registry_uri or _default_registry_uri("Production"),
        "scaler": joblib.load(os.path.join(art_dir, "scaler.pkl")),
        "global_threshold": float(np.load(os.path.join(art_dir, "global_threshold.npy"))),
        "env_thresholds": np.load(os.path.join(art_dir, "env_thresholds.npy")),
        "sensor_threshold": float(np.load(os.path.join(art_dir, "sensor_threshold.npy"))),
        "normal_pf": (normal_pf := np.load(
            os.path.join(art_dir, "normal_errors_per_feature.npy")
        )),
        "val_p99": compute_val_p99(normal_pf),
        "meta": meta,
    }


def build_stream_timeline(
    data_path: str,
    artifacts_dir: str,
    models_dir: str,
    window_filter: str = "drift",
    registry_uri: str | None = None,
) -> dict[str, Any]:
    """
    Precompute per-timestep arrays for simulated real-time replay.

    Returns dict with df, feature_error (T,F), predicted_drift, drift_type (coarse),
    drift_type_fine, feature_names, env_idxs, sensor_idx, thresholds.
    """
    uri = registry_uri or _default_registry_uri("Production")
    bundle = load_artifacts(artifacts_dir, models_dir, registry_uri=uri)
    model_source = bundle.get("model_source", "local")
    meta = bundle["meta"]
    all_features: list[str] = meta["all_features"]
    env_idxs: list[int] = meta["env_indices"]
    sensor_idx: int = meta["sensor_index"]
    window_size: int = meta["window_size"]

    df = pd.read_csv(data_path, parse_dates=["DATEPRD"])
    if window_filter:
        df = df[df["WINDOW"] == window_filter].copy().reset_index(drop=True)

    x_raw = df[all_features].values.astype(np.float32)
    x_scaled = bundle["scaler"].transform(x_raw)
    x_seq = make_sequences(x_scaled, window_size)
    x_pred = bundle["model"].predict(x_seq, verbose=0)
    errors_per_feat = recon_error_per_feature(x_seq, x_pred)

    val_p99 = bundle["val_p99"]
    fine_labels, _, _, _, s_pos, _, norm_errors = classify_windows(
        errors_per_feat=errors_per_feat,
        val_p99=val_p99,
        all_features=all_features,
        env_idxs=env_idxs,
        sensor_idx=sensor_idx,
    )

    t_rows = len(df)
    n_feat = len(all_features)
    pad = window_size - 1

    feature_error = np.full((t_rows, n_feat), np.nan, dtype=np.float64)
    norm_error_timeline = np.full((t_rows, n_feat), np.nan, dtype=np.float64)
    cusum_timeline = np.full((t_rows, n_feat), np.nan, dtype=np.float64)
    drift_type_fine = np.array(["No Drift"] * t_rows, dtype=object)
    drift_type = np.array(["No Drift"] * t_rows, dtype=object)
    predicted_drift = np.zeros(t_rows, dtype=bool)

    for i, label in enumerate(fine_labels):
        t = pad + i
        feature_error[t] = errors_per_feat[i]
        norm_error_timeline[t] = norm_errors[i]
        cusum_timeline[t] = s_pos[i]
        drift_type_fine[t] = label
        coarse = coarsen_label(label)
        drift_type[t] = coarse
        predicted_drift[t] = coarse != "No Drift"

    return {
        "df": df,
        "feature_error": feature_error,
        "norm_errors": norm_error_timeline,
        "cusum_s_pos": cusum_timeline,
        "val_p99": val_p99,
        "predicted_drift": predicted_drift,
        "drift_type": drift_type,
        "drift_type_fine": drift_type_fine,
        "feature_names": all_features,
        "env_idxs": env_idxs,
        "sensor_idx": sensor_idx,
        "sensor_threshold": bundle["sensor_threshold"],
        "env_thresholds": bundle["env_thresholds"],
        "global_threshold": bundle["global_threshold"],
        "window_size": window_size,
        "source": data_path,
        "classifier_version": "v8_fixed_val_p99",
        "detector_config": {
            "cusum_k": CUSUM_K,
            "cusum_h": CUSUM_H,
            "normalisation": "fixed_val_p99",
            "max_normalized_error": MAX_NORMALIZED_ERROR,
            "roll_z_window": ROLL_Z_WINDOW,
            "roll_z_thresh": ROLL_Z_THRESH,
            "persist": PERSIST,
            "annulus_dominance_thresh": ANNULUS_DOMINANCE_THRESH,
            "env_dominance_thresh": ENV_DOMINANCE_THRESH,
        },
        "model_source": model_source,
        "mlflow_model_uri": uri,
        "registry_run_id": bundle.get("registry_run_id"),
        "registry_version_stage": uri.split("/")[-1] if uri.startswith("models:/") else None,
    }


def build_drift_explanation(
    *,
    predicted_drift: bool,
    row_err: np.ndarray,
    norm_row: np.ndarray,
    cusum_row: np.ndarray,
    val_p99: np.ndarray,
    coarse_type: str,
    fine_type: str,
    feature_names: list[str],
    env_idxs: list[int],
    sensor_idx: int,
    sensor_threshold: float,
) -> list[dict[str, Any]]:
    """Feature-level explanations aligned with v8 fixed val_p99 + CUSUM detection."""
    if not predicted_drift or np.isnan(row_err).any():
        return []

    explanation: list[dict[str, Any]] = []

    def _entry(
        feat_idx: int,
        *,
        reason: str,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        display = DISPLAY_NAMES.get(feature_names[feat_idx], feature_names[feat_idx])
        entry: dict[str, Any] = {
            "feature_name": feature_names[feat_idx],
            "display_name": display,
            "error": float(row_err[feat_idx]),
            "baseline": float(val_p99[feat_idx]),
            "norm_error": float(norm_row[feat_idx]) if not np.isnan(norm_row[feat_idx]) else None,
            "cusum": float(cusum_row[feat_idx]) if not np.isnan(cusum_row[feat_idx]) else None,
            "reason": reason,
        }
        if threshold is not None:
            entry["threshold"] = float(threshold)
        return entry

    if coarse_type == "Sensor Drift":
        for i in env_idxs:
            explanation.append(
                _entry(
                    i,
                    reason="Environmental context (stable during annulus sensor fault)",
                )
            )
        explanation.append(
            _entry(
                sensor_idx,
                reason=(
                    f"Annulus CUSUM dominance (k={CUSUM_K}, h={CUSUM_H}); "
                    f"fine label: {fine_type}"
                ),
                threshold=sensor_threshold,
            )
        )
        return explanation

    for i in env_idxs:
        norm = norm_row[i]
        cusum = cusum_row[i]
        if fine_type.startswith("Probable Sensor Fault:"):
            reason = (
                f"Elevated env CUSUM — possible single-sensor fault "
                f"(norm={norm:.2f}x val_p99, S+={cusum:.1f})"
            )
        else:
            reason = (
                f"Fixed val_p99 normalised error {norm:.2f}x "
                f"(ref=1.0); CUSUM S+={cusum:.1f}"
            )
        explanation.append(_entry(i, reason=reason))

    return explanation
