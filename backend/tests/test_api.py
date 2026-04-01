import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add backend module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Mock all heavy files BEFORE importing backend ────────
dummy_df = pd.DataFrame({
    "DATEPRD": pd.date_range("2020-01-01", periods=100),
    "AVG_DOWNHOLE_PRESSURE": np.random.rand(100),
    "AVG_DOWNHOLE_TEMPERATURE": np.random.rand(100),
    "BORE_OIL_VOL": np.random.rand(100),
    "AVG_WHP_P": np.random.rand(100),
    "DP_CHOKE_SIZE": np.random.rand(100),
    "window": ["baseline"] * 30 + ["normal"] * 50 + ["drift"] * 20
})

dummy_array = np.zeros(100)
dummy_2d = np.zeros((100, 5))

with patch("pandas.read_excel", return_value=dummy_df), \
     patch("numpy.load", return_value=dummy_array), \
     patch("joblib.load", return_value=MagicMock()), \
     patch("tensorflow.keras.models.load_model", return_value=MagicMock()):
    from backend import app

from fastapi.testclient import TestClient
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

# ── Test 4: Out of bounds returns done ───────────────────
def test_drift_data_out_of_bounds():
    response = client.get("/drift_data?t=999999")
    assert response.status_code == 200
    assert response.json()["done"] == True