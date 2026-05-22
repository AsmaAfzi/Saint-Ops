"""
Shared SAINT inference: LSTM reconstruction + rolling p99 CUSUM + dominance-ratio classification (v7).
Used by test_model.py (evaluation) and backend.py (stream precompute).
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ── Detector / classifier parameters (v6) ─────────────────────────────────────

CUSUM_K = 0.5
CUSUM_H = 4.0
ROLL_P99_WINDOW = 50
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


def rolling_p99_normalize(
    errors_per_feat: np.ndarray, window: int, cap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise errors by rolling p99 per feature; cap at `cap` multiples of p99."""
    df = pd.DataFrame(errors_per_feat)
    roll_p99 = (
        df.rolling(window, min_periods=10).quantile(0.99).fillna(df.quantile(0.99))
    )
    roll_p99_safe = roll_p99.clip(lower=1e-8)
    normalized = (df / roll_p99_safe).clip(upper=cap)
    return normalized.values, roll_p99.values


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
    np.ndarray,
]:
    norm_errors, roll_p99 = rolling_p99_normalize(
        errors_per_feat, ROLL_P99_WINDOW, MAX_NORMALIZED_ERROR
    )
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
        env_ratio = leader_s / others_mean

        if env_ratio > ENV_DOMINANCE_THRESH and ann_s < max_env:
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
        roll_p99,
    )


def load_artifacts(artifacts_dir: str, models_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(artifacts_dir, "feature_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "model": load_model(os.path.join(models_dir, "lstm_autoencoder.keras")),
        "scaler": joblib.load(os.path.join(artifacts_dir, "scaler.pkl")),
        "global_threshold": float(np.load(os.path.join(artifacts_dir, "global_threshold.npy"))),
        "env_thresholds": np.load(os.path.join(artifacts_dir, "env_thresholds.npy")),
        "sensor_threshold": float(np.load(os.path.join(artifacts_dir, "sensor_threshold.npy"))),
        "normal_pf": np.load(os.path.join(artifacts_dir, "normal_errors_per_feature.npy")),
        "meta": meta,
    }


def build_stream_timeline(
    data_path: str,
    artifacts_dir: str,
    models_dir: str,
    window_filter: str = "drift",
) -> dict[str, Any]:
    """
    Precompute per-timestep arrays for simulated real-time replay.

    Returns dict with df, feature_error (T,F), predicted_drift, drift_type (coarse),
    drift_type_fine, feature_names, env_idxs, sensor_idx, thresholds.
    """
    bundle = load_artifacts(artifacts_dir, models_dir)
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

    fine_labels, _, _, _, _, _, _, _ = classify_windows(
        errors_per_feat=errors_per_feat,
        all_features=all_features,
        env_idxs=env_idxs,
        sensor_idx=sensor_idx,
    )

    t_rows = len(df)
    n_feat = len(all_features)
    pad = window_size - 1

    feature_error = np.full((t_rows, n_feat), np.nan, dtype=np.float64)
    drift_type_fine = np.array(["No Drift"] * t_rows, dtype=object)
    drift_type = np.array(["No Drift"] * t_rows, dtype=object)
    predicted_drift = np.zeros(t_rows, dtype=bool)

    for i, label in enumerate(fine_labels):
        t = pad + i
        feature_error[t] = errors_per_feat[i]
        drift_type_fine[t] = label
        coarse = coarsen_label(label)
        drift_type[t] = coarse
        predicted_drift[t] = coarse != "No Drift"

    return {
        "df": df,
        "feature_error": feature_error,
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
        "classifier_version": "v7_rolling_p99",
    }
