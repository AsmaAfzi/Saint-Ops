import os
import random
import sys

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_root_dir() -> str:
    """
    Resolve project root for data/artifacts/models paths.

    - Docker: backend.py and volume mounts live under /app
    - Local dev: backend/ is a subfolder; data/ is next to backend/
    """
    if os.path.isdir(os.path.join(BACKEND_DIR, "data")):
        return BACKEND_DIR
    parent = os.path.dirname(BACKEND_DIR)
    if os.path.isdir(os.path.join(parent, "data")):
        return parent
    return BACKEND_DIR


def _setup_inference_import() -> None:
    """
    Import shared inference + registry from ml/ (local) or /app/ (Docker bundle).
    """
    bundled_inference = os.path.join(BACKEND_DIR, "inference.py")
    if os.path.isfile(bundled_inference):
        if BACKEND_DIR not in sys.path:
            sys.path.insert(0, BACKEND_DIR)
        return
    ml_dir = os.path.join(_resolve_root_dir(), "ml")
    if os.path.isfile(os.path.join(ml_dir, "inference.py")) and ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)


ROOT_DIR = _resolve_root_dir()
_setup_inference_import()

from inference import build_stream_timeline, _default_registry_uri  # noqa: E402

try:
    from registry import (  # noqa: E402
        approve_staging,
        compare_staging_vs_production,
        list_version_history,
        promote_staging_to_production,
        rollback_production,
    )
except ImportError:
    approve_staging = None  # type: ignore[misc, assignment]
    compare_staging_vs_production = None
    list_version_history = None
    promote_staging_to_production = None
    rollback_production = None

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
_shadow_stream: dict | None = None
_load_error: str | None = None
_shadow_load_error: str | None = None

PRODUCTION_URI = os.environ.get(
    "MLFLOW_MODEL_URI", _default_registry_uri("Production")
)
SHADOW_URI = os.environ.get(
    "MLFLOW_SHADOW_MODEL_URI", _default_registry_uri("Staging")
)
SHADOW_TRAFFIC_PCT = float(os.environ.get("MLFLOW_SHADOW_TRAFFIC_PCT", "0"))


def invalidate_streams() -> None:
    global _stream, _shadow_stream, _load_error, _shadow_load_error
    _stream = None
    _shadow_stream = None
    _load_error = None
    _shadow_load_error = None


def get_stream(*, registry_uri: str | None = None) -> dict:
    global _stream, _load_error
    uri = registry_uri or PRODUCTION_URI
    if registry_uri is None and _stream is not None:
        return _stream
    if registry_uri is None and _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        timeline = build_stream_timeline(
            data_path=STREAM_CSV,
            artifacts_dir=ARTIFACTS_DIR,
            models_dir=MODELS_DIR,
            window_filter="drift",
            registry_uri=uri,
        )
        if registry_uri is None:
            _stream = timeline
        return timeline
    except Exception as exc:  # noqa: BLE001
        if registry_uri is None:
            _load_error = str(exc)
        raise


def get_shadow_stream() -> dict:
    global _shadow_stream, _shadow_load_error
    if _shadow_stream is not None:
        return _shadow_stream
    if _shadow_load_error is not None:
        raise RuntimeError(_shadow_load_error)
    try:
        _shadow_stream = build_stream_timeline(
            data_path=STREAM_CSV,
            artifacts_dir=ARTIFACTS_DIR,
            models_dir=MODELS_DIR,
            window_filter="drift",
            registry_uri=SHADOW_URI,
        )
        return _shadow_stream
    except Exception as exc:  # noqa: BLE001
        _shadow_load_error = str(exc)
        raise


def _resolve_stream_variant(
    variant: str | None,
    x_shadow_traffic: str | None,
) -> tuple[dict, str]:
    """Pick production or shadow stream (gradual traffic split via header/env)."""
    if variant == "shadow":
        return get_shadow_stream(), "shadow"
    if variant == "production":
        return get_stream(), "production"

    pct = SHADOW_TRAFFIC_PCT
    if x_shadow_traffic is not None:
        try:
            pct = float(x_shadow_traffic)
        except ValueError:
            pass

    if pct > 0 and random.random() * 100.0 < pct:
        try:
            return get_shadow_stream(), "shadow"
        except RuntimeError:
            return get_stream(), "production"
    return get_stream(), "production"


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
def get_drift_data(
    t: int,
    variant: str | None = Query(
        None,
        description="production | shadow (omit for gradual split via MLFLOW_SHADOW_TRAFFIC_PCT)",
    ),
    x_shadow_traffic: str | None = Header(None, alias="X-Shadow-Traffic-Pct"),
):
    try:
        stream, served_variant = _resolve_stream_variant(variant, x_shadow_traffic)
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
        "served_variant": served_variant,
        "mlflow_model_uri": stream.get("mlflow_model_uri"),
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
        shadow_ready = False
        shadow_error: str | None = None
        try:
            get_shadow_stream()
            shadow_ready = True
        except RuntimeError as exc:
            shadow_error = str(exc)

        return {
            "status": "ready",
            "artifacts_loaded": True,
            "drift_data_available": True,
            "stream_rows": len(stream["df"]),
            "stream_source": os.path.basename(stream["source"]),
            "model_source": stream.get("model_source", "local"),
            "mlflow_model_uri": stream.get("mlflow_model_uri"),
            "shadow_model_uri": SHADOW_URI,
            "shadow_ready": shadow_ready,
            "shadow_error": shadow_error,
            "shadow_traffic_pct": SHADOW_TRAFFIC_PCT,
        }
    except RuntimeError as exc:
        return {
            "status": "not_ready",
            "artifacts_loaded": False,
            "drift_data_available": False,
            "error": str(exc),
        }


class PromoteRequest(BaseModel):
    force: bool = False


@app.get("/models/info")
def models_info():
    if list_version_history is None:
        return {"error": "registry module unavailable"}
    return {
        "registered_model": os.environ.get("MLFLOW_REGISTERED_MODEL", "SAINT"),
        "production_uri": PRODUCTION_URI,
        "shadow_uri": SHADOW_URI,
        "shadow_traffic_pct": SHADOW_TRAFFIC_PCT,
        "production": get_stream().get("mlflow_model_uri"),
        "versions": list_version_history(limit=10),
    }


@app.get("/models/compare")
def models_compare():
    if compare_staging_vs_production is None:
        return {"error": "registry module unavailable"}
    return compare_staging_vs_production()


@app.get("/models/shadow/compare")
def shadow_compare_at_t(t: int = Query(0, ge=0)):
    """Side-by-side production vs staging predictions at timestep t."""
    try:
        prod = get_stream()
        shadow = get_shadow_stream()
    except RuntimeError as exc:
        return {"error": str(exc)}

    if t >= len(prod["df"]):
        return {"done": True}

    prod_err = prod["feature_error"][t]
    shadow_err = shadow["feature_error"][t]
    return {
        "time": t,
        "production": {
            "uri": prod.get("mlflow_model_uri"),
            "drift_type": str(prod["drift_type"][t]),
            "predicted_drift": bool(prod["predicted_drift"][t]),
            "feature_error": np.nan_to_num(prod_err, nan=0.0).tolist(),
        },
        "shadow": {
            "uri": shadow.get("mlflow_model_uri"),
            "drift_type": str(shadow["drift_type"][t]),
            "predicted_drift": bool(shadow["predicted_drift"][t]),
            "feature_error": np.nan_to_num(shadow_err, nan=0.0).tolist(),
        },
        "agreement": str(prod["drift_type"][t]) == str(shadow["drift_type"][t]),
    }


@app.post("/models/approve")
def models_approve():
    if approve_staging is None:
        return {"ok": False, "error": "registry module unavailable"}
    return approve_staging()


@app.post("/models/promote")
def models_promote(body: PromoteRequest):
    if promote_staging_to_production is None:
        return {"ok": False, "error": "registry module unavailable"}
    result = promote_staging_to_production(force=body.force)
    if result.get("ok"):
        invalidate_streams()
    return result


@app.post("/models/rollback")
def models_rollback():
    if rollback_production is None:
        return {"ok": False, "error": "registry module unavailable"}
    result = rollback_production()
    if result.get("ok"):
        invalidate_streams()
    return result


@app.post("/models/reload")
def models_reload():
    """Hot-reload production + shadow streams from registry (no container restart)."""
    invalidate_streams()
    try:
        stream = get_stream()
        shadow_ok = False
        try:
            get_shadow_stream()
            shadow_ok = True
        except RuntimeError:
            pass
        return {
            "ok": True,
            "model_source": stream.get("model_source"),
            "mlflow_model_uri": stream.get("mlflow_model_uri"),
            "shadow_ready": shadow_ok,
        }
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
