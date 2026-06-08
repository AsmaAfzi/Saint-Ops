import os
import sys

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend as backend_module

N = 100
dummy_df = pd.DataFrame({
    "DATEPRD": pd.date_range("2020-01-01", periods=N),
})

backend_module._stream = {
    "df": dummy_df,
    "feature_error": np.random.rand(N, 5).astype(np.float64),
    "predicted_drift": np.array([False] * 20 + [True] * 80),
    "drift_type": np.array(["No Drift"] * 20 + ["Sensor Drift"] * 80, dtype=object),
    "drift_type_fine": np.array(["No Drift"] * 20 + ["Sensor Fault: Annulus Pressure Gauge"] * 80, dtype=object),
    "feature_names": [
        "AVG_DOWNHOLE_PRESSURE",
        "AVG_DOWNHOLE_TEMPERATURE",
        "BORE_OIL_VOL",
        "AVG_WHP_P",
        "AVG_ANNULUS_PRESS",
    ],
    "env_idxs": [0, 1, 2, 3],
    "sensor_idx": 4,
    "sensor_threshold": 0.05,
    "env_thresholds": np.array([0.01, 0.01, 0.01, 0.01]),
    "global_threshold": 0.1,
    "window_size": 10,
    "source": "mock",
    "model_source": "local",
    "mlflow_model_uri": "models:/SAINT/Production",
}

from backend import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_drift_data_structure():
    response = client.get("/drift_data?t=25")
    assert response.status_code == 200
    data = response.json()
    assert "time" in data
    assert data["feature_names"] == backend_module._stream["feature_names"]
    assert len(data["feature_error"]) == 5


def test_drift_data_out_of_bounds():
    response = client.get("/drift_data?t=999999")
    assert response.status_code == 200
    assert response.json()["done"] is True


def test_models_reload(monkeypatch):
    dummy = backend_module._stream

    def _fake_timeline(**_kwargs):
        return dummy

    monkeypatch.setattr(backend_module, "build_stream_timeline", _fake_timeline)
    response = client.post("/models/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "mlflow_model_uri" in data


def test_models_info():
    response = client.get("/models/info")
    assert response.status_code == 200
    body = response.json()
    assert "production_uri" in body or "error" in body


def test_drift_data_served_variant():
    response = client.get("/drift_data?t=25&variant=production")
    assert response.status_code == 200
    data = response.json()
    assert data.get("served_variant") == "production"
