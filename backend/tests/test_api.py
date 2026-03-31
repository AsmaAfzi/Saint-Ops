import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
import pandas as pd
import sys
import os

# Add backend module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Mock pandas and numpy data before importing backend ─────────
dummy_df = pd.DataFrame({
    "DATEPRD": pd.date_range("2026-01-01", periods=5),
    "feature1": [0,1,2,3,4],
    "feature2": [5,6,7,8,9]
})

dummy_feature_error = np.random.rand(5,2)
dummy_predicted_drift = np.array([False]*5)
dummy_drift_type = np.array(["Sensor Drift"]*5)

with patch("backend.pd.read_excel", return_value=dummy_df), \
     patch("backend.np.load", side_effect=[dummy_feature_error, dummy_predicted_drift, dummy_drift_type]):
    from backend import app

client = TestClient(app)

# ── Test 1: Health endpoint ──────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# ── Test 2: Ready endpoint ───────────────────────────────
def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200

# ── Test 3: Drift data returns valid structure ────────────
def test_drift_data_structure():
    response = client.get("/drift_data?t=0")
    assert response.status_code == 200
    data = response.json()
    assert "time" in data or "done" in data

# ── Test 4: Drift data at valid time index ────────────────
def test_drift_data_valid_index():
    response = client.get("/drift_data?t=100")
    assert response.status_code == 200
    data = response.json()
    if not data.get("done"):
        assert "drift_type" in data
        assert "sensor_error" in data
        assert "env_error" in data
        assert "feature_names" in data

# ── Test 5: Drift data beyond dataset returns done ────────
def test_drift_data_out_of_bounds():
    response = client.get("/drift_data?t=999999")
    assert response.status_code == 200
    assert response.json()["done"] == True