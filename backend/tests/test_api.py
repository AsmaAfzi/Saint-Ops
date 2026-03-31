import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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