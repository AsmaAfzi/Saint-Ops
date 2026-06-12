"""
Poll SAINT backend for sustained environmental drift and trigger retraining DAG.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime

from airflow import DAG
from airflow.api.common.trigger_dag import trigger_dag
from airflow.operators.python import PythonOperator

TRIGGER_DIR = os.environ.get("RETRAIN_TRIGGER_DIR", "/opt/saint-retrain")
TRIGGER_FILE = os.path.join(TRIGGER_DIR, "trigger.json")
BACKEND_URL = os.environ.get("SAINT_BACKEND_URL", "http://backend:8000")


def check_drift_and_maybe_retrain() -> None:
    with urllib.request.urlopen(f"{BACKEND_URL}/ops/drift-summary", timeout=30) as resp:
        summary = json.loads(resp.read().decode())
    if not summary.get("retrain_candidate"):
        return
    os.makedirs(TRIGGER_DIR, exist_ok=True)
    if not os.path.isfile(TRIGGER_FILE):
        payload = {
            "reason": "consecutive_env_drift",
            "triggered_at": datetime.utcnow().isoformat(),
            "consecutive_env_drift": summary.get("consecutive_env_drift_windows"),
        }
        with open(TRIGGER_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    trigger_dag(dag_id="saint_retrain", run_id=None, replace_microseconds=False)


with DAG(
    dag_id="saint_drift_sensor",
    description="Trigger retrain when environmental drift persists",
    schedule_interval="*/15 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["saint-ops", "drift"],
) as dag:
    PythonOperator(
        task_id="check_drift_and_trigger",
        python_callable=check_drift_and_maybe_retrain,
    )
