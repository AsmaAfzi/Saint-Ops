"""
Manual MLflow snapshot (optional — train/eval already log automatically).

Usage:
    docker compose up mlflow -d
    cd ml
    python log_mlflow.py
"""

from __future__ import annotations

import os

from mlflow_log import log_full_snapshot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

if __name__ == "__main__":
    log_full_snapshot(
        artifacts_dir=os.path.join(ROOT_DIR, "artifacts"),
        models_dir=os.path.join(ROOT_DIR, "models"),
        results_dir=os.path.join(BASE_DIR, "results"),
    )
    print("Open http://localhost:5000 → experiment SAINT-OPS")
