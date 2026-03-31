# ============================================================
# Robust Drift Detection + Explainability + Live Plots - LSTM
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
WINDOW_SIZE = 10
PERSIST_SENSOR = 4
PERSIST_ENV = 3

ROLL_BASELINE = 50
ENV_DEV_MULT = 1.5
TOP_K_FEATURES = 3

SENSOR_FEATURES = ["DP_CHOKE_SIZE", "AVG_DOWNHOLE_PRESSURE"]
ENV_FEATURES = [
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "BORE_OIL_VOL",
    "AVG_WHP_P"
]

# -----------------------------
# File paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/synthetic_drift_dataset_v2.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "../models/lstm_autoencoder_baseline.keras")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler_lstm.pkl")
DRIFT_THRESHOLD_PATH = os.path.join(BASE_DIR, "../artifacts/drift_threshold.npy")
SENSOR_THRESHOLD_PATH = os.path.join(BASE_DIR, "../artifacts/sensor_thresholds.npy")
ENV_THRESHOLD_PATH = os.path.join(BASE_DIR, "../artifacts/env_thresholds.npy")

# -----------------------------
# Helpers
# -----------------------------
def make_sequences(X, w):
    return np.array([X[i:i+w] for i in range(len(X) - w + 1)])

def persist(flags, k):
    return pd.Series(flags).rolling(k, min_periods=1).sum() >= k

# -----------------------------
# Load artifacts
# -----------------------------
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
global_threshold = np.load(DRIFT_THRESHOLD_PATH)
sensor_thresholds = np.load(SENSOR_THRESHOLD_PATH)
env_thresholds = np.load(ENV_THRESHOLD_PATH)

FEATURES = list(scaler.feature_names_in_)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel(DATA_PATH)
df = df.sort_values("DATEPRD").reset_index(drop=True)
true_drift = (df["window"] == "drift").astype(int)

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()

X_scaled = scaler.transform(X)
X_seq = make_sequences(X_scaled, WINDOW_SIZE)
pad = WINDOW_SIZE - 1

# -----------------------------
# Reconstruction
# -----------------------------
X_pred = model.predict(X_seq, verbose=0)

recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))
recon_error = np.concatenate([np.full(pad, np.nan), recon_error])

feature_error = np.mean(np.square(X_seq - X_pred), axis=1)
feature_error = np.concatenate([np.full((pad, feature_error.shape[1]), np.nan), feature_error])

# -----------------------------
# Sensor drift
# -----------------------------
sensor_flags = np.zeros(len(df), dtype=bool)
for i, f in enumerate(FEATURES):
    if f in SENSOR_FEATURES:
        idx = FEATURES.index(f)
        sensor_flags |= feature_error[:, idx] > sensor_thresholds[SENSOR_FEATURES.index(f)]
sensor_flags = persist(sensor_flags, PERSIST_SENSOR)

# -----------------------------
# Environmental drift
# -----------------------------
env_idxs = [FEATURES.index(f) for f in ENV_FEATURES]

env_baseline = (
    pd.DataFrame(feature_error[:, env_idxs])
    .rolling(ROLL_BASELINE, min_periods=20)
    .median()
    .to_numpy()
)

env_relative_error = np.nanmean(
    np.abs(feature_error[:, env_idxs] - env_baseline),
    axis=1
)

env_trend = pd.Series(env_relative_error).diff() > 0
env_flags_raw = env_relative_error > (global_threshold * ENV_DEV_MULT)
env_flags = persist(env_flags_raw & env_trend, PERSIST_ENV)

# -----------------------------
# Final prediction
# -----------------------------
predicted_drift = sensor_flags | env_flags
drift_type = np.full(len(df), "No Drift", dtype=object)
drift_type[sensor_flags & ~env_flags] = "Sensor Drift"
drift_type[env_flags] = "Environmental Drift"

# -----------------------------
# Evaluation
# -----------------------------
print("=== Binary Drift Detection ===")
print(confusion_matrix(true_drift, predicted_drift.astype(int)))
print(classification_report(true_drift, predicted_drift.astype(int), zero_division=0))

print("\n=== Multi-Class Drift Detection ===")
mapping = {"No Drift": 0, "Sensor Drift": 1, "Environmental Drift": 2}
y_true = np.where(true_drift == 1, 2, 0)
y_pred = pd.Series(drift_type).map(mapping)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=mapping.keys(), zero_division=0))

# ============================================================
# 🔍 Drift Explainability
# ============================================================
explanations = []

for t in range(len(df)):
    if not predicted_drift[t] or np.isnan(feature_error[t]).any():
        continue
    errors = feature_error[t]
    ranked = np.argsort(errors)[::-1][:TOP_K_FEATURES]
    explanations.append({
        "Time_Index": t,
        "Drift_Type": drift_type[t],
        "Top_Features": [FEATURES[i] for i in ranked],
        "Errors": [errors[i] for i in ranked]
    })

explain_df = pd.DataFrame(explanations)
print("\n=== Drift Explainability (sample) ===")
print(explain_df.head())
explain_df.to_csv(os.path.join(BASE_DIR, "../artifacts/drift_explanations.csv"), index=False)

# ============================================================
# 📈 Per-feature Reconstruction Plots (Live)
# ============================================================
X_pred_point = X_pred[:, -1, :]
X_pred_point = np.vstack([np.full((pad, X_pred_point.shape[1]), np.nan), X_pred_point])

for i, feature in enumerate(FEATURES):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=X_scaled[:, i], mode="lines", name="True", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=X_pred_point[:, i], mode="lines", name="Reconstructed", line=dict(color="orange")))
    drift_idx = np.where(predicted_drift)[0]
    fig.add_trace(go.Scatter(x=drift_idx, y=X_scaled[drift_idx, i], mode="markers", name="Detected Drift", marker=dict(color="red", size=6)))
    fig.update_layout(title=f"Reconstruction Plot — {feature}", xaxis_title="Time Index", yaxis_title="Scaled Value", template="plotly_white")
    fig.show()

# ============================================================
# 📊 Global Reconstruction Error Plot
# ============================================================
fig = px.scatter(x=range(len(recon_error)), y=recon_error, color=drift_type,
                 labels={"x": "Time Index", "y": "Reconstruction Error"},
                 title="Global Reconstruction Error with Drift Labels")
fig.add_hline(y=global_threshold, line_dash="dash", line_color="red")
fig.show()