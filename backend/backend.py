import json
import os
import random
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from pydantic import BaseModel

REQUEST_COUNT = Counter(
    "saint_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
INFERENCE_LATENCY = Histogram(
    "saint_inference_latency_seconds",
    "Inference endpoint latency (drift_data, ready)",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
DRIFT_DETECTIONS = Counter(
    "saint_drift_detections_total",
    "Drift events served via /drift_data",
    ["drift_type", "variant"],
)
RECONSTRUCTION_ERROR = Histogram(
    "saint_reconstruction_error",
    "Per-request reconstruction error (sensor vs environmental mean)",
    ["channel"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
MODEL_INFO = Info("saint_model_version", "Active production model metadata")
CONSECUTIVE_ENV_DRIFT = Gauge(
    "saint_consecutive_env_drift_windows",
    "Consecutive environmental drift windows in the live stream",
)
RETRAIN_JOBS = Counter(
    "saint_retrain_jobs_total",
    "Retraining pipeline runs",
    ["status"],
)

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

from inference import (  # noqa: E402
    build_drift_explanation,
    build_stream_timeline,
    _default_registry_uri,
)

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

@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        import asyncio

        await asyncio.to_thread(get_stream)
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title="SAINT-OPS Backend", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    if path in ("/drift_data", "/ready", "/models/shadow/compare"):
        INFERENCE_LATENCY.labels(endpoint=path).observe(elapsed)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=path,
        status=str(response.status_code),
    ).inc()
    return response

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(ROOT_DIR, "artifacts"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT_DIR, "models"))
STREAM_CSV = os.environ.get(
    "STREAM_CSV", os.path.join(DATA_DIR, "scenario_C_annulus.csv")
)

TOP_K = 3

_stream: dict | None = None
_shadow_stream: dict | None = None
_load_error: str | None = None
_shadow_load_error: str | None = None
_consecutive_env_drift: int = 0
_alert_events: list[dict] = []
_retrain_state: dict = {
    "last_trigger": None,
    "last_status": "idle",
    "last_message": None,
}

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
            _publish_model_info(timeline)
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


def _publish_model_info(stream: dict) -> None:
    MODEL_INFO.info(
        {
            "uri": str(stream.get("mlflow_model_uri") or PRODUCTION_URI),
            "source": str(stream.get("model_source") or "local"),
            "classifier": str(stream.get("classifier_version") or "unknown"),
        }
    )


def _record_drift_metrics(
    *,
    coarse_type: str,
    served_variant: str,
    sensor_err: float,
    env_err: float,
    predicted_drift: bool,
) -> None:
    global _consecutive_env_drift
    RECONSTRUCTION_ERROR.labels(channel="sensor").observe(sensor_err)
    RECONSTRUCTION_ERROR.labels(channel="env").observe(env_err)
    if predicted_drift:
        DRIFT_DETECTIONS.labels(drift_type=coarse_type, variant=served_variant).inc()
    if coarse_type == "Environmental Drift" and predicted_drift:
        _consecutive_env_drift += 1
    else:
        _consecutive_env_drift = 0
    CONSECUTIVE_ENV_DRIFT.set(_consecutive_env_drift)


def _safe_float(value: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)


def _build_explanation(
    stream: dict, t: int, row_err: np.ndarray, coarse_type: str, fine_type: str
) -> list[dict]:
    norm_row = stream.get("norm_errors")
    cusum_row = stream.get("cusum_s_pos")
    val_p99 = stream.get("val_p99")
    if norm_row is None or cusum_row is None or val_p99 is None:
        return []

    return build_drift_explanation(
        predicted_drift=bool(stream["predicted_drift"][t]),
        row_err=row_err,
        norm_row=norm_row[t],
        cusum_row=cusum_row[t],
        val_p99=val_p99,
        coarse_type=coarse_type,
        fine_type=fine_type,
        feature_names=stream["feature_names"],
        env_idxs=stream["env_idxs"],
        sensor_idx=stream["sensor_idx"],
        sensor_threshold=float(stream["sensor_threshold"]),
    )


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
    fine_type = str(stream["drift_type_fine"][t])
    norm_row = stream.get("norm_errors")
    cusum_row = stream.get("cusum_s_pos")

    sensor_err = _safe_float(row_err[sensor_idx])
    env_err = float(np.nanmean(row_err[env_idxs])) if not np.isnan(row_err[env_idxs]).all() else 0.0
    predicted_drift = bool(stream["predicted_drift"][t])
    _record_drift_metrics(
        coarse_type=coarse_type,
        served_variant=served_variant,
        sensor_err=sensor_err,
        env_err=env_err,
        predicted_drift=predicted_drift,
    )

    return {
        "time": t,
        "datetime": str(df.loc[t, "DATEPRD"]),
        "sensor_error": sensor_err,
        "env_error": env_err,
        "predicted_drift": predicted_drift,
        "drift_type": coarse_type,
        "drift_type_fine": fine_type,
        "feature_error": np.nan_to_num(row_err, nan=0.0).tolist(),
        "feature_norm_error": (
            np.nan_to_num(norm_row[t], nan=0.0).tolist() if norm_row is not None else []
        ),
        "feature_cusum": (
            np.nan_to_num(cusum_row[t], nan=0.0).tolist() if cusum_row is not None else []
        ),
        "feature_names": stream["feature_names"],
        "explanation": _build_explanation(stream, t, row_err, coarse_type, fine_type),
        "classifier_version": stream.get("classifier_version", "v8_fixed_val_p99"),
        "detector_config": stream.get("detector_config"),
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
            "classifier_version": stream.get("classifier_version"),
            "detector_config": stream.get("detector_config"),
        }
    except RuntimeError as exc:
        return {"error": str(exc)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SAINT-OPS Backend",
        "model": "LSTM Autoencoder",
        "version": "1.3.0",
        "classifier": "v8_fixed_val_p99",
    }


@app.get("/metrics")
def metrics():
    """Prometheus scrape target (HPA custom metrics via prometheus-adapter)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ready")
def ready():
    try:
        stream = get_stream()
        # Avoid eager shadow timeline build on every /ready (very expensive).
        shadow_ready = _shadow_stream is not None
        shadow_error: str | None = _shadow_load_error

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
            "classifier_version": stream.get("classifier_version"),
            "detector_config": stream.get("detector_config"),
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
    try:
        return {
            "registered_model": os.environ.get("MLFLOW_REGISTERED_MODEL", "SAINT"),
            "production_uri": PRODUCTION_URI,
            "shadow_uri": SHADOW_URI,
            "shadow_traffic_pct": SHADOW_TRAFFIC_PCT,
            "production": get_stream().get("mlflow_model_uri"),
            "versions": list_version_history(limit=10),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@app.get("/models/compare")
def models_compare():
    if compare_staging_vs_production is None:
        return {"error": "registry module unavailable"}
    try:
        return compare_staging_vs_production()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


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

    def _variant_payload(stream: dict, timestep: int) -> dict:
        err = stream["feature_error"][timestep]
        norm = stream.get("norm_errors")
        cusum = stream.get("cusum_s_pos")
        return {
            "uri": stream.get("mlflow_model_uri"),
            "drift_type": str(stream["drift_type"][timestep]),
            "drift_type_fine": str(stream["drift_type_fine"][timestep]),
            "predicted_drift": bool(stream["predicted_drift"][timestep]),
            "classifier_version": stream.get("classifier_version"),
            "detector_config": stream.get("detector_config"),
            "feature_error": np.nan_to_num(err, nan=0.0).tolist(),
            "feature_norm_error": (
                np.nan_to_num(norm[timestep], nan=0.0).tolist() if norm is not None else []
            ),
            "feature_cusum": (
                np.nan_to_num(cusum[timestep], nan=0.0).tolist() if cusum is not None else []
            ),
        }

    prod_payload = _variant_payload(prod, t)
    shadow_payload = _variant_payload(shadow, t)
    return {
        "time": t,
        "production": prod_payload,
        "shadow": shadow_payload,
        "agreement": prod_payload["drift_type"] == shadow_payload["drift_type"],
        "agreement_fine": prod_payload["drift_type_fine"] == shadow_payload["drift_type_fine"],
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


@app.post("/ops/alerts/webhook")
async def ops_alerts_webhook(request: Request):
    """Alertmanager receiver — stores recent alerts for demo UI (no external SaaS)."""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    entry = {"received_at": time.time(), "payload": payload}
    _alert_events.append(entry)
    _alert_events[:] = _alert_events[-50:]
    return {"ok": True}


@app.get("/ops/alerts")
def ops_alerts_recent(limit: int = Query(20, ge=1, le=50)):
    return {"alerts": list(reversed(_alert_events[-limit:]))}


@app.get("/ops/drift-summary")
def ops_drift_summary():
    threshold = int(os.environ.get("RETRAIN_ENV_DRIFT_WINDOWS", "50"))
    return {
        "consecutive_env_drift_windows": _consecutive_env_drift,
        "retrain_threshold_windows": threshold,
        "retrain_candidate": _consecutive_env_drift >= threshold,
        "retrain_state": _retrain_state,
    }


class RetrainTriggerRequest(BaseModel):
    reason: str = "manual"


@app.get("/retrain/status")
def retrain_status():
    compare: dict | None = None
    if compare_staging_vs_production is not None:
        try:
            compare = compare_staging_vs_production()
        except Exception:  # noqa: BLE001
            compare = None
    return {
        "state": _retrain_state,
        "drift_summary": ops_drift_summary(),
        "registry_compare": compare,
    }


@app.post("/retrain/trigger")
def retrain_trigger(body: RetrainTriggerRequest):
    """Signal Airflow / local trigger file for demo retraining pipeline."""
    return _do_retrain_trigger(body.reason)


@app.get("/retrain/trigger/demo")
def retrain_trigger_demo():
    """One-click demo trigger (open in browser). Local demo only."""
    return _do_retrain_trigger("demo")


def _do_retrain_trigger(reason: str):
    trigger_dir = os.environ.get("RETRAIN_TRIGGER_DIR", "/tmp/saint-retrain")
    os.makedirs(trigger_dir, exist_ok=True)
    trigger_file = os.path.join(trigger_dir, "trigger.json")
    payload = {
        "reason": reason,
        "triggered_at": time.time(),
        "consecutive_env_drift": _consecutive_env_drift,
    }
    with open(trigger_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    _retrain_state.update(
        {
            "last_trigger": payload["triggered_at"],
            "last_status": "queued",
            "last_message": reason,
        }
    )
    RETRAIN_JOBS.labels(status="queued").inc()
    return {"ok": True, "trigger_file": trigger_file, "payload": payload}


@app.post("/retrain/complete")
def retrain_complete(status: str = "success", message: str = ""):
    """Called by Airflow DAG on completion (demo callback)."""
    _retrain_state.update(
        {
            "last_status": status,
            "last_message": message or status,
            "completed_at": time.time(),
        }
    )
    RETRAIN_JOBS.labels(status=status).inc()
    return {"ok": True, "state": _retrain_state}


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
