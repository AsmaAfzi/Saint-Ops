# ============================================================
# LSTM Autoencoder Training for Drift Detection (FINAL)
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# CONFIG
# -----------------------------
WINDOW_SIZE = 10

FEATURES = [
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "BORE_OIL_VOL",
    "AVG_WHP_P",
    "DP_CHOKE_SIZE"
]

# Define environment and sensor features separately
ENV_FEATURES = [
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "BORE_OIL_VOL",
    "AVG_WHP_P"
]

SENSOR_FEATURES = [
    "DP_CHOKE_SIZE",
    "AVG_DOWNHOLE_PRESSURE"  # sensor-relevant features only
]

# -----------------------------
# File path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/synthetic_drift_dataset_v2.xlsx")

# -----------------------------
# Load baseline data only
# -----------------------------
df = pd.read_excel(DATA_PATH)
df = df[df["window"] == "baseline"]

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).dropna()

# -----------------------------
# Scale
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(BASE_DIR, "../models/scaler_lstm.pkl"))

# -----------------------------
# Sliding windows
# -----------------------------
def make_sequences(X, w):
    return np.array([X[i:i+w] for i in range(len(X) - w + 1)])

X_seq = make_sequences(X_scaled, WINDOW_SIZE)
print("Training shape:", X_seq.shape)

# -----------------------------
# LSTM Autoencoder
# -----------------------------
n_features = X_seq.shape[2]

inputs = Input(shape=(WINDOW_SIZE, n_features))
encoded = LSTM(64, activation="relu")(inputs)
latent = Dense(32, activation="relu")(encoded)
decoded = RepeatVector(WINDOW_SIZE)(latent)
decoded = LSTM(64, activation="relu", return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(n_features))(decoded)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3), loss="mse")
model.summary()

# -----------------------------
# Train
# -----------------------------
model.fit(
    X_seq, X_seq,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

model.save(os.path.join(BASE_DIR, "../models/lstm_autoencoder_baseline.keras"))

# -----------------------------
# Thresholds
# -----------------------------
X_pred = model.predict(X_seq, verbose=0)

# Global threshold (binary drift)
recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))
global_threshold = np.percentile(recon_error, 95)
np.save(os.path.join(BASE_DIR, "../artifacts/drift_threshold.npy"), global_threshold)

# Per-feature thresholds
feature_error = np.mean(np.square(X_seq - X_pred), axis=1)
sensor_idxs = [FEATURES.index(f) for f in SENSOR_FEATURES]
sensor_thresholds = np.percentile(feature_error[:, sensor_idxs], 95, axis=0)
np.save(os.path.join(BASE_DIR, "../artifacts/sensor_thresholds.npy"), sensor_thresholds)

# Environmental thresholds (per env feature)
env_idxs = [FEATURES.index(f) for f in ENV_FEATURES]
env_thresholds = np.percentile(feature_error[:, env_idxs], 95, axis=0)
np.save(os.path.join(BASE_DIR, "../artifacts/env_thresholds.npy"), env_thresholds)

print("✔ Training complete")
print("Global threshold:", global_threshold)
print("Sensor thresholds:", sensor_thresholds)
print("Environmental thresholds:", env_thresholds)