import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ARTIFACTS_DIR = os.path.join(BASE_DIR, "../artifacts")
#DATA_DIR = os.path.join(BASE_DIR, "../data")

BASE_DIR = "/app"
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Load data
df = pd.read_excel(os.path.join(DATA_DIR, "synthetic_drift_dataset_v2.xlsx"))
feature_error = np.load(os.path.join(ARTIFACTS_DIR, "feature_error.npy"))
predicted_drift = np.load(os.path.join(ARTIFACTS_DIR, "predicted_drift.npy"))
drift_type = np.load(os.path.join(ARTIFACTS_DIR, "drift_type.npy"), allow_pickle=True)

# FEATURES used by model (exclude DATEPRD)
MODEL_FEATURES = [f for f in df.columns if f != "DATEPRD"]
TOP_K = 3

@app.get("/drift_data")
def get_drift_data(t: int):
    if t >= len(df):
        return {"done": True}

    row_err = feature_error[t]
    explanation = []

    if predicted_drift[t]:
        if drift_type[t] == "Sensor Drift":
            for i, val in enumerate(row_err):
                explanation.append({
                    "feature_name": MODEL_FEATURES[i],
                    "error": float(val),
                    "threshold": 0.1,  # adjust according to sensor_thresholds
                    "reason": "Exceeded sensor threshold"
                })
        else:
            baseline = np.nanmean(feature_error[max(0, t-50):t], axis=0)
            for i, val in enumerate(row_err):
                explanation.append({
                    "feature_name": MODEL_FEATURES[i],
                    "error": float(val),
                    "baseline": float(baseline[i]),
                    "reason": "Deviation increasing from baseline"
                })

    return {
        "time": t,
        "datetime": str(df.loc[t, "DATEPRD"]),
        "sensor_error": float(np.nanmax(row_err)),
        "env_error": float(np.nanmean(row_err)),
        "predicted_drift": bool(predicted_drift[t]),
        "drift_type": str(drift_type[t]),
        "feature_error": row_err.tolist(),
        "feature_names": MODEL_FEATURES,
        "explanation": explanation,
        "done": False
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SAINT-OPS Backend",
        "model": "LSTM Autoencoder",
        "version": "1.0.0"
    }

@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "artifacts_loaded": True,
        "drift_data_available": True
    }

# -----------------------------
# Run server (for testing)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
