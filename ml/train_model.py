"""
SAINT — LSTM Autoencoder Training Script
Input:  f12_clean.csv   (output of prepare_f12_data.py)

Outputs:
    models/lstm_autoencoder.keras
    artifacts/scaler.pkl
    artifacts/global_threshold.npy
    artifacts/env_thresholds.npy        shape (4,) — one per env feature
    artifacts/sensor_threshold.npy
    artifacts/normal_errors.npy
    artifacts/normal_errors_per_feature.npy
    artifacts/feature_meta.json

CHANGES FROM v1:
    - AVG_DP_TUBING removed from ENV_FEATURES.
      Model now takes 5 features as input (4 env + 1 sensor) instead of 6.
    - env_idxs updated to reflect 4 env features (indices 0-3).
    - sensor_idx updated to 4 (was 5).
    - env_thresholds shape is now (4,) not (5,).
    All architecture and training hyperparameters unchanged.

Usage:
    python train_saint.py

Requirements:
    pip install tensorflow scikit-learn pandas numpy joblib
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense,
                                     RepeatVector, TimeDistributed, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "../data/f12_clean.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "../models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "../artifacts")

# CHANGED: AVG_DP_TUBING removed. 4 env features, 1 sensor feature = 5 total.
ENV_FEATURES = [
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "BORE_OIL_VOL",
    "AVG_WHP_P",
]
SENSOR_FEATURES = [
    "AVG_ANNULUS_PRESS",
]
ALL_FEATURES = ENV_FEATURES + SENSOR_FEATURES   # 5 features, order is fixed

WINDOW_SIZE = 10
STEP_SIZE   = 1

LSTM_UNITS   = 64
LATENT_DIM   = 16
DROPOUT      = 0.2

LEARNING_RATE    = 1e-3
BATCH_SIZE       = 16
MAX_EPOCHS       = 200
PATIENCE         = 15
LR_PATIENCE      = 7
VALIDATION_SPLIT = 0.15
THRESHOLD_PERCENTILE = 99

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def make_sequences(data: np.ndarray, window_size: int,
                   step_size: int = 1) -> np.ndarray:
    """Convert (T, F) array into overlapping (N, window_size, F) sequences."""
    sequences = []
    for start in range(0, len(data) - window_size + 1, step_size):
        sequences.append(data[start : start + window_size])
    return np.array(sequences)


def reconstruction_error_per_window(X_true: np.ndarray,
                                    X_pred: np.ndarray) -> np.ndarray:
    """MSE per window averaged across timesteps and features. Shape: (N,)."""
    return np.mean(np.square(X_true - X_pred), axis=(1, 2))


def reconstruction_error_per_feature(X_true: np.ndarray,
                                     X_pred: np.ndarray) -> np.ndarray:
    """MSE per window per feature averaged across timesteps. Shape: (N, F)."""
    return np.mean(np.square(X_true - X_pred), axis=1)


# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

print("=" * 60)
print("SAINT — LSTM Autoencoder Training")
print("=" * 60)

print(f"\n[1] Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, parse_dates=["DATEPRD"])
print(f"    Total rows: {len(df)}")

baseline_df = df[df["WINDOW"] == "baseline"][ALL_FEATURES].copy()
normal_df   = df[df["WINDOW"] == "normal"][ALL_FEATURES].copy()

print(f"    Baseline rows : {len(baseline_df)}  (training data only)")
print(f"    Normal rows   : {len(normal_df)}  (threshold calibration only)")
print(f"    Features ({len(ALL_FEATURES)})  : {ALL_FEATURES}")

assert baseline_df.isnull().sum().sum() == 0, \
    "Nulls in baseline — re-run prepare_f12_data.py"
assert normal_df.isnull().sum().sum() == 0, \
    "Nulls in normal window — re-run prepare_f12_data.py"

# ── STEP 2: SCALE ─────────────────────────────────────────────────────────────
# Scaler FIT on baseline only. Normal window is only transformed, never fit.

print(f"\n[2] Fitting StandardScaler on baseline ...")
scaler = StandardScaler()
baseline_scaled = scaler.fit_transform(baseline_df.values)
normal_scaled   = scaler.transform(normal_df.values)

print(f"    Scaled baseline — mean: {baseline_scaled.mean():.4f}  "
      f"std: {baseline_scaled.std():.4f}")

# ── STEP 3: SEQUENCES ─────────────────────────────────────────────────────────

print(f"\n[3] Creating sliding window sequences "
      f"(window={WINDOW_SIZE}, step={STEP_SIZE}) ...")
X_train  = make_sequences(baseline_scaled, WINDOW_SIZE, STEP_SIZE)
X_normal = make_sequences(normal_scaled,   WINDOW_SIZE, STEP_SIZE)

print(f"    Training sequences : {X_train.shape}")
print(f"    Normal sequences   : {X_normal.shape}")

# ── STEP 4: BUILD MODEL ───────────────────────────────────────────────────────
# Encoder: LSTM(64, tanh) → Dropout → Dense(16) bottleneck
# Decoder: RepeatVector → LSTM(64, tanh) → Dropout → TimeDistributed Dense(5)
#
# tanh (not relu) for LSTM activations: LSTM gates use sigmoid/tanh internally;
# relu in recurrent layers causes gradient issues and dead neurons.
# Latent dim 16: small bottleneck forces strong compression, increases
# sensitivity to features that break the learned inter-feature correlations.

print(f"\n[4] Building LSTM autoencoder ...")
n_features = len(ALL_FEATURES)   # 5

inputs = Input(shape=(WINDOW_SIZE, n_features), name="encoder_input")

enc    = LSTM(LSTM_UNITS, activation="tanh",
              return_sequences=False, name="encoder_lstm")(inputs)
enc    = Dropout(DROPOUT, name="encoder_dropout")(enc)
latent = Dense(LATENT_DIM, activation="relu", name="latent")(enc)

dec    = RepeatVector(WINDOW_SIZE, name="repeat_vector")(latent)
dec    = LSTM(LSTM_UNITS, activation="tanh",
              return_sequences=True, name="decoder_lstm")(dec)
dec    = Dropout(DROPOUT, name="decoder_dropout")(dec)
output = TimeDistributed(Dense(n_features), name="output")(dec)

model = Model(inputs, output, name="SAINT_LSTM_AE")
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
model.summary()

# ── STEP 5: TRAIN ─────────────────────────────────────────────────────────────

print(f"\n[5] Training ...")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=LR_PATIENCE, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, X_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)

stopped_epoch = len(history.history["loss"])
print(f"\n    Stopped at epoch {stopped_epoch}")
print(f"    Train loss : {history.history['loss'][-1]:.6f}")
print(f"    Val loss   : {history.history['val_loss'][-1]:.6f}")

# ── STEP 6: CALIBRATE THRESHOLDS ON NORMAL WINDOW ────────────────────────────
# Model has NEVER seen the normal window during training.
# p99 of normal-window errors → threshold that 99% of normal operation falls under.

print(f"\n[6] Calibrating thresholds on normal window ...")
X_normal_pred = model.predict(X_normal, verbose=0)

normal_errors_global  = reconstruction_error_per_window(X_normal, X_normal_pred)
normal_errors_per_feat = reconstruction_error_per_feature(X_normal, X_normal_pred)

global_threshold = np.percentile(normal_errors_global, THRESHOLD_PERCENTILE)

# CHANGED: env_idxs is now 0-3 (4 features), sensor_idx is 4 (was 5).
env_idxs       = list(range(len(ENV_FEATURES)))   # [0, 1, 2, 3]
sensor_idx     = len(ENV_FEATURES)                 # 4

env_thresholds   = np.percentile(
    normal_errors_per_feat[:, env_idxs],
    THRESHOLD_PERCENTILE, axis=0)                  # shape (4,)
sensor_threshold = np.percentile(
    normal_errors_per_feat[:, sensor_idx],
    THRESHOLD_PERCENTILE)

print(f"    Global threshold (p{THRESHOLD_PERCENTILE}): {global_threshold:.6f}")
print(f"    Normal error range: [{normal_errors_global.min():.6f}, "
      f"{normal_errors_global.max():.6f}]")
print(f"\n    Per-feature thresholds (p{THRESHOLD_PERCENTILE}):")
for i, feat in enumerate(ENV_FEATURES):
    print(f"      {feat:<35} {env_thresholds[i]:.6f}")
print(f"      {'AVG_ANNULUS_PRESS':<35} {sensor_threshold:.6f}")

# ── STEP 7: SAVE ARTIFACTS ────────────────────────────────────────────────────

print(f"\n[7] Saving artifacts ...")
os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

model_path  = os.path.join(MODEL_DIR,    "lstm_autoencoder.keras")
scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
gt_path     = os.path.join(ARTIFACT_DIR, "global_threshold.npy")
et_path     = os.path.join(ARTIFACT_DIR, "env_thresholds.npy")
st_path     = os.path.join(ARTIFACT_DIR, "sensor_threshold.npy")
ne_path     = os.path.join(ARTIFACT_DIR, "normal_errors.npy")
npf_path    = os.path.join(ARTIFACT_DIR, "normal_errors_per_feature.npy")
meta_path   = os.path.join(ARTIFACT_DIR, "feature_meta.json")

model.save(model_path)
joblib.dump(scaler, scaler_path)
np.save(gt_path,  global_threshold)
np.save(et_path,  env_thresholds)
np.save(st_path,  sensor_threshold)
np.save(ne_path,  normal_errors_global)
np.save(npf_path, normal_errors_per_feat)

feature_meta = {
    "all_features":    ALL_FEATURES,
    "env_features":    ENV_FEATURES,
    "sensor_features": SENSOR_FEATURES,
    "env_indices":     env_idxs,
    "sensor_index":    sensor_idx,
    "window_size":     WINDOW_SIZE,
}
with open(meta_path, "w") as f:
    json.dump(feature_meta, f, indent=2)

print(f"    Model              → {model_path}")
print(f"    Scaler             → {scaler_path}")
print(f"    Global threshold   → {gt_path}")
print(f"    Env thresholds     → {et_path}   shape: {env_thresholds.shape}")
print(f"    Sensor threshold   → {st_path}")
print(f"    Normal errors      → {ne_path}   shape: {normal_errors_global.shape}")
print(f"    Normal per-feat    → {npf_path}  shape: {normal_errors_per_feat.shape}")
print(f"    Feature metadata   → {meta_path}")

# ── STEP 8: SANITY CHECK ──────────────────────────────────────────────────────

print(f"\n[8] Sanity check — baseline reconstruction errors ...")
X_train_pred = model.predict(X_train, verbose=0)
train_errors = reconstruction_error_per_window(X_train, X_train_pred)

print(f"    Baseline error mean : {train_errors.mean():.6f}")
print(f"    Baseline error max  : {train_errors.max():.6f}")
print(f"    Global threshold    : {global_threshold:.6f}")

if train_errors.max() < global_threshold:
    print(f"    ✓ All baseline errors below threshold — model learned correctly")
else:
    pct = (train_errors > global_threshold).mean() * 100
    print(f"    ⚠ {pct:.1f}% of baseline windows exceed threshold")
    print(f"      Try increasing DROPOUT or reducing LSTM_UNITS and retrain.")

print(f"\n{'='*60}")
print(f"Training complete.")
print(f"Next step: python evaluate_saint.py")
print(f"{'='*60}")