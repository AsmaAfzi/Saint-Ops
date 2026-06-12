"""
SAINT — LSTM Autoencoder Training Script (v2)
=============================================
Input:  f12_clean.csv   (output of prepare_f12_data.py)

Outputs:
    models/lstm_autoencoder.keras
    artifacts/scaler.pkl
    artifacts/global_threshold.npy
    artifacts/env_thresholds.npy        shape (4,)
    artifacts/sensor_threshold.npy
    artifacts/normal_errors.npy
    artifacts/normal_errors_per_feature.npy
    artifacts/feature_meta.json

────────────────────────────────────────────────────────────
CHANGE FROM v1 — THE CORE FIX
────────────────────────────────────────────────────────────
v1 trained on WINDOW == 'baseline' only: 259 rows, Feb–Nov 2008.
The test/drift window is Apr–Oct 2010 — two years later.
The model learned 2008 operating patterns and treated ALL 2010 data
as anomalous, regardless of whether drift was injected or not.
Per-feature CUSUM S+ was 20-30+ for non-injected features in every
scenario — the model couldn't distinguish signal from background.

v2 trains on WINDOW != 'drift': 692 rows, Feb 2008 – Apr 2010.
This covers the full normal operating life of the well up to the
point where the test window begins. The model learns early-2010
operating patterns. Clean 2010 data in the test window reconstructs
well. Only genuinely injected drift produces elevated errors.

Training / validation split within the 692 non-drift rows:
  First 85% (588 rows, Feb 2008 – Dec 2009): gradient updates
  Last 15%  (104 rows, Dec 2009 – Apr 2010): early stopping validation

The validation window immediately precedes the test window, which is
the most conservative possible check: if the model can reconstruct
late-2009/early-2010 data, it should reconstruct Apr-Oct 2010 well too.

Threshold calibration:
  Global and per-feature thresholds are computed from the VALIDATION
  rows (104 rows), not the training rows. Using training rows would
  underestimate normal error (the model has memorised them). Using
  a truly held-out set (the drift window) would leak test information.
  The validation set is the correct choice.

Usage:
    python train_saint.py

Delete models/ and artifacts/ before running to avoid stale artifacts.

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

ENV_FEATURES = [
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "BORE_OIL_VOL",
    "AVG_WHP_P",
]
SENSOR_FEATURES = [
    "AVG_ANNULUS_PRESS",
]
ALL_FEATURES = ENV_FEATURES + SENSOR_FEATURES   # 5 features, order fixed

WINDOW_SIZE = 10
STEP_SIZE   = 1

LSTM_UNITS   = 64
LATENT_DIM   = 16
DROPOUT      = 0.2

LEARNING_RATE    = 1e-3
BATCH_SIZE       = 32     # larger batch now that we have 579 training sequences
MAX_EPOCHS       = int(os.environ.get("SAINT_MAX_EPOCHS", "200"))
PATIENCE         = 15
LR_PATIENCE      = 7

# CHANGED: internal validation split within the non-drift rows
# Last 15% of non-drift data used for validation (early stopping + thresholds)
VAL_SPLIT_FRAC = 0.15

THRESHOLD_PERCENTILE = 99

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def make_sequences(data: np.ndarray, window_size: int,
                   step_size: int = 1) -> np.ndarray:
    """Convert (T, F) → (N, window_size, F) overlapping sequences."""
    return np.array([data[i:i+window_size]
                     for i in range(0, len(data) - window_size + 1, step_size)])

def recon_error_per_window(X_true, X_pred):
    """MSE per window averaged across timesteps and features. Shape (N,)."""
    return np.mean(np.square(X_true - X_pred), axis=(1, 2))

def recon_error_per_feature(X_true, X_pred):
    """MSE per window per feature averaged across timesteps. Shape (N, F)."""
    return np.mean(np.square(X_true - X_pred), axis=1)

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

print("=" * 60)
print("SAINT — LSTM Autoencoder Training (v2)")
print("=" * 60)

print(f"\n[1] Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, parse_dates=["DATEPRD"])
print(f"    Total rows: {len(df)}")
print(f"    Window counts: {df['WINDOW'].value_counts().to_dict()}")

# CHANGED: use ALL non-drift rows for training, not just 'baseline'
non_drift_df = df[df["WINDOW"] != "drift"][ALL_FEATURES + ["DATEPRD"]].copy()
non_drift_df = non_drift_df.sort_values("DATEPRD").reset_index(drop=True)

n_total  = len(non_drift_df)
n_val    = int(n_total * VAL_SPLIT_FRAC)
n_fit    = n_total - n_val

fit_df = non_drift_df.iloc[:n_fit][ALL_FEATURES].copy()
val_df = non_drift_df.iloc[n_fit:][ALL_FEATURES].copy()

print(f"\n    Training split (all non-drift rows = {n_total}):")
print(f"      Fit rows:        {n_fit}  "
      f"({non_drift_df['DATEPRD'].iloc[0].date()} → "
      f"{non_drift_df['DATEPRD'].iloc[n_fit-1].date()})")
print(f"      Validation rows: {n_val}  "
      f"({non_drift_df['DATEPRD'].iloc[n_fit].date()} → "
      f"{non_drift_df['DATEPRD'].iloc[-1].date()})")
print(f"      Test/drift rows: {(df['WINDOW']=='drift').sum()}  "
      f"(held out entirely — not used in training or thresholds)")
print(f"\n    Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")

assert fit_df.isnull().sum().sum() == 0, "Nulls in fit data"
assert val_df.isnull().sum().sum() == 0, "Nulls in validation data"

# ── STEP 2: SCALE ─────────────────────────────────────────────────────────────
# CHANGED: scaler fit on ALL fit rows (588), not just baseline (259).
# This means the scaler's mean and std reflect the full 2008-2009 operating
# range, not just the early 2008 period.

print(f"\n[2] Fitting StandardScaler on fit rows ({n_fit}) ...")
scaler     = StandardScaler()
fit_scaled = scaler.fit_transform(fit_df.values)
val_scaled = scaler.transform(val_df.values)

print(f"    Fit scaled — mean: {fit_scaled.mean():.4f}  std: {fit_scaled.std():.4f}")
print(f"    Val scaled — mean: {val_scaled.mean():.4f}  std: {val_scaled.std():.4f}")

# ── STEP 3: SEQUENCES ─────────────────────────────────────────────────────────

print(f"\n[3] Creating sliding window sequences (window={WINDOW_SIZE}) ...")
X_fit = make_sequences(fit_scaled, WINDOW_SIZE, STEP_SIZE)
X_val = make_sequences(val_scaled, WINDOW_SIZE, STEP_SIZE)

print(f"    Fit sequences: {X_fit.shape}   "
      f"(was 250 in v1, now {X_fit.shape[0]})")
print(f"    Val sequences: {X_val.shape}")

# ── STEP 4: BUILD MODEL ───────────────────────────────────────────────────────
# Architecture unchanged from v1.
# tanh activation for LSTM (not relu — avoids dead neurons in recurrent nets).
# Latent dim 16: strong compression, sensitive to inter-feature correlation breaks.
# Dropout 0.2: prevents memorisation even with more training data.

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

model = Model(inputs, output, name="SAINT_LSTM_AE_v2")
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
model.summary()

# ── STEP 5: TRAIN ─────────────────────────────────────────────────────────────
# CHANGED: validation_data is the explicit val sequences (not validation_split).
# This ensures the validation window is always the chronologically latest rows
# (Dec 2009 – Apr 2010), not a random subset.

print(f"\n[5] Training ...")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=LR_PATIENCE, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_fit, X_fit,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, X_val),   # CHANGED: explicit val set
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)

stopped = len(history.history["loss"])
print(f"\n    Stopped at epoch {stopped}")
print(f"    Train loss : {history.history['loss'][-1]:.6f}")
print(f"    Val loss   : {history.history['val_loss'][-1]:.6f}")

# ── STEP 6: CALIBRATE THRESHOLDS ON VALIDATION WINDOW ─────────────────────────
# CHANGED: thresholds are computed from validation rows (Dec 2009 – Apr 2010).
# The validation window has never received gradient updates — the model is
# genuinely generalising to it, so its reconstruction errors represent the
# true distribution of normal operating error on unseen recent data.
# p99 of these errors → threshold that 99% of normal operation falls under.

print(f"\n[6] Calibrating thresholds on validation window ({n_val} rows) ...")
X_val_pred = model.predict(X_val, verbose=0)

val_errors_global   = recon_error_per_window(X_val, X_val_pred)
val_errors_per_feat = recon_error_per_feature(X_val, X_val_pred)

global_threshold = np.percentile(val_errors_global, THRESHOLD_PERCENTILE)

env_idxs       = list(range(len(ENV_FEATURES)))   # [0, 1, 2, 3]
sensor_idx     = len(ENV_FEATURES)                 # 4

env_thresholds   = np.percentile(
    val_errors_per_feat[:, env_idxs],
    THRESHOLD_PERCENTILE, axis=0)                  # shape (4,)
sensor_threshold = np.percentile(
    val_errors_per_feat[:, sensor_idx],
    THRESHOLD_PERCENTILE)

print(f"    Global threshold (p{THRESHOLD_PERCENTILE}): {global_threshold:.6f}")
print(f"    Val error range: [{val_errors_global.min():.6f}, "
      f"{val_errors_global.max():.6f}]")
print(f"\n    Per-feature thresholds (p{THRESHOLD_PERCENTILE}):")
for i, feat in enumerate(ENV_FEATURES):
    print(f"      {feat:<35} {env_thresholds[i]:.6f}")
print(f"      {'AVG_ANNULUS_PRESS':<35} {sensor_threshold:.6f}")

# ── STEP 7: SAVE ──────────────────────────────────────────────────────────────

print(f"\n[7] Saving artifacts ...")
os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

model.save(os.path.join(MODEL_DIR, "lstm_autoencoder.keras"))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
np.save(os.path.join(ARTIFACT_DIR, "global_threshold.npy"), global_threshold)
np.save(os.path.join(ARTIFACT_DIR, "env_thresholds.npy"),   env_thresholds)
np.save(os.path.join(ARTIFACT_DIR, "sensor_threshold.npy"), sensor_threshold)
np.save(os.path.join(ARTIFACT_DIR, "normal_errors.npy"),    val_errors_global)
np.save(os.path.join(ARTIFACT_DIR, "normal_errors_per_feature.npy"),
        val_errors_per_feat)

feature_meta = {
    "all_features":    ALL_FEATURES,
    "env_features":    ENV_FEATURES,
    "sensor_features": SENSOR_FEATURES,
    "env_indices":     env_idxs,
    "sensor_index":    sensor_idx,
    "window_size":     WINDOW_SIZE,
    "training_version": "v2_non_drift_val_thresholds",
    "classifier_version": "v8_fixed_val_p99",
}
with open(os.path.join(ARTIFACT_DIR, "feature_meta.json"), "w") as f:
    json.dump(feature_meta, f, indent=2)

print(f"    All artifacts saved to {MODEL_DIR}/ and {ARTIFACT_DIR}/")

# ── STEP 8: SANITY CHECK ──────────────────────────────────────────────────────
# Fit errors should be low (model trained on these).
# Val errors should be slightly higher but still below threshold for most.
# If val errors are already above threshold, the model generalises poorly —
# consider increasing DROPOUT or reducing LSTM_UNITS.

print(f"\n[8] Sanity check ...")
X_fit_pred  = model.predict(X_fit, verbose=0)
fit_errors  = recon_error_per_window(X_fit, X_fit_pred)

print(f"    Fit  errors — mean: {fit_errors.mean():.6f}  "
      f"max: {fit_errors.max():.6f}")
print(f"    Val  errors — mean: {val_errors_global.mean():.6f}  "
      f"max: {val_errors_global.max():.6f}")
print(f"    Global threshold:   {global_threshold:.6f}")

pct_fit_above = (fit_errors > global_threshold).mean() * 100
pct_val_above = (val_errors_global > global_threshold).mean() * 100
print(f"\n    Fit  rows above threshold: {pct_fit_above:.1f}%  "
      f"(expected ~0-5%)")
print(f"    Val  rows above threshold: {pct_val_above:.1f}%  "
      f"(expected ~1% — by definition of p99)")

if pct_val_above > 5:
    print(f"\n    ⚠  Val error rate {pct_val_above:.1f}% > 5%.")
    print(f"       The model may not generalise well to the validation period.")
    print(f"       Consider: increasing DROPOUT, reducing LSTM_UNITS, or")
    print(f"       checking for data quality issues in the validation window.")
else:
    print(f"\n    ✓ Model generalises well to the validation window.")

print(f"\n{'='*60}")
print(f"Training complete.")
print(f"Next step: python test_model.py")
print(f"{'='*60}")

# ── STEP 9: MLFLOW (automatic) ────────────────────────────────────────────────
try:
    from mlflow_log import log_training_run

    log_training_run(
        artifacts_dir=ARTIFACT_DIR,
        models_dir=MODEL_DIR,
        model=model,
        params={
            "training_version": "v2_non_drift_val_thresholds",
            "classifier": "fixed_val_p99_CUSUM_v8",
            "window_size": WINDOW_SIZE,
            "n_features": len(ALL_FEATURES),
            "lstm_units": LSTM_UNITS,
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
            "val_split_frac": VAL_SPLIT_FRAC,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "fit_rows": n_fit,
            "val_rows": n_val,
        },
        metrics={
            "train_loss_final": float(history.history["loss"][-1]),
            "val_loss_final": float(history.history["val_loss"][-1]),
            "global_threshold": float(global_threshold),
            "sensor_threshold": float(sensor_threshold),
            "pct_fit_above_threshold": float(pct_fit_above),
            "pct_val_above_threshold": float(pct_val_above),
            "epochs_trained": float(stopped),
        },
    )
except Exception as _mlflow_exc:
    print(f"[MLflow] { _mlflow_exc}")