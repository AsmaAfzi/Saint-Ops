import os
import sys

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Project root + ml package for shared inference
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
ML_DIR = os.path.join(ROOT_DIR, "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from inference import build_stream_timeline  # noqa: E402

app = FastAPI(title="SAINT-OPS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(ROOT_DIR, "artifacts"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT_DIR, "models"))
STREAM_CSV = os.environ.get(
    "STREAM_CSV", os.path.join(DATA_DIR, "scenario_C_annulus.csv")
)

TOP_K = 3
ROLL_BASELINE = 50

_stream: dict | None = None
_load_error: str | None = None


def get_stream() -> dict:
    global _stream, _load_error
    if _stream is not None:
        return _stream
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        _stream = build_stream_timeline(
            data_path=STREAM_CSV,
            artifacts_dir=ARTIFACTS_DIR,
            models_dir=MODELS_DIR,
            window_filter="drift",
        )
        return _stream
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        raise


def _safe_float(value: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)


def _build_explanation(
    stream: dict, t: int, row_err: np.ndarray, coarse_type: str
) -> list[dict]:
    if not stream["predicted_drift"][t] or np.isnan(row_err).any():
        return []

    env_idxs = stream["env_idxs"]
    sensor_idx = stream["sensor_idx"]
    feature_names = stream["feature_names"]
    explanation: list[dict] = []

    if coarse_type == "Sensor Drift":
        for i in env_idxs:
            explanation.append(
                {
                    "feature_name": feature_names[i],
                    "error": _safe_float(row_err[i]),
                    "baseline": _safe_float(row_err[i]),
                    "reason": "Environmental context (stable during sensor fault)",
                }
            )
        explanation.append(
            {
                "feature_name": feature_names[sensor_idx],
                "error": _safe_float(row_err[sensor_idx]),
                "threshold": stream["sensor_threshold"],
                "reason": "Exceeded annulus sensor threshold",
            }
        )
    else:
        baseline = np.nanmean(stream["feature_error"][max(0, t - ROLL_BASELINE) : t], axis=0)
        for j, i in enumerate(env_idxs):
            explanation.append(
                {
                    "feature_name": feature_names[i],
                    "error": _safe_float(row_err[i]),
                    "baseline": _safe_float(baseline[i]),
                    "threshold": _safe_float(stream["env_thresholds"][j]),
                    "reason": "Deviation from rolling baseline",
                }
            )

    return explanation


@app.get("/drift_data")
def get_drift_data(t: int):
    try:
        stream = get_stream()
    except RuntimeError as exc:
        return {"done": True, "error": str(exc)}

    df = stream["df"]
    if t >= len(df):
        return {"done": True}

    row_err = stream["feature_error"][t]
    env_idxs = stream["env_idxs"]
    sensor_idx = stream["sensor_idx"]
    coarse_type = str(stream["drift_type"][t])

    sensor_err = _safe_float(row_err[sensor_idx])
    env_err = float(np.nanmean(row_err[env_idxs])) if not np.isnan(row_err[env_idxs]).all() else 0.0

    return {
        "time": t,
        "datetime": str(df.loc[t, "DATEPRD"]),
        "sensor_error": sensor_err,
        "env_error": env_err,
        "predicted_drift": bool(stream["predicted_drift"][t]),
        "drift_type": coarse_type,
        "drift_type_fine": str(stream["drift_type_fine"][t]),
        "feature_error": np.nan_to_num(row_err, nan=0.0).tolist(),
        "feature_names": stream["feature_names"],
        "explanation": _build_explanation(stream, t, row_err, coarse_type),
        "done": False,
    }


@app.get("/stream/info")
def stream_info():
    try:
        stream = get_stream()
        return {
            "length": len(stream["df"]),
            "source": stream["source"],
            "window_filter": "drift",
            "features": stream["feature_names"],
            "window_size": stream["window_size"],
        }
    except RuntimeError as exc:
        return {"error": str(exc)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SAINT-OPS Backend",
        "model": "LSTM Autoencoder",
        "version": "1.1.0",
    }


@app.get("/ready")
def ready():
    try:
        stream = get_stream()
        return {
            "status": "ready",
            "artifacts_loaded": True,
            "drift_data_available": True,
            "stream_rows": len(stream["df"]),
            "stream_source": os.path.basename(stream["source"]),
        }
    except RuntimeError as exc:
        return {
            "status": "not_ready",
            "artifacts_loaded": False,
            "drift_data_available": False,
            "error": str(exc),
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
