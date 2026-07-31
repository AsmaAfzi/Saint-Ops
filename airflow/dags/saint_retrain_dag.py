"""
SAINT-OPS automated retraining DAG (Phase 6).

Triggers: manual (Airflow UI), weekly schedule, or saint_drift_sensor DAG.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

REPO_ROOT = os.environ.get("SAINT_REPO_ROOT", "/opt/saint")
ARTIFACTS = os.path.join(REPO_ROOT, "artifacts")
TRIGGER_DIR = os.environ.get("RETRAIN_TRIGGER_DIR", "/opt/saint-retrain")
TRIGGER_FILE = os.path.join(TRIGGER_DIR, "trigger.json")
BACKEND_URL = os.environ.get("SAINT_BACKEND_URL", "http://backend:8000")
MIN_F1 = float(os.environ.get("RETRAIN_MIN_F1", "0.80"))

default_args = {
    "owner": "saint-ops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _parse_eval_f1() -> float:
    report = os.path.join(REPO_ROOT, "ml", "results", "evaluation_report.txt")
    if not os.path.isfile(report):
        return 0.0
    with open(report, encoding="utf-8") as f:
        text = f.read()
    for pattern in (
        r"Sensor-pathway F1 \(Scen\. B\)\s*:\s*([\d.]+)",
        r"Binary drift-presence F1\s*:\s*([\d.]+)",
        r"Headline global F1 \(mean\)\s*:\s*([\d.]+)",
        r"Macro F1 \(coarse\):\s*([\d.]+)",
    ):
        import re

        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return 0.0


def quality_gate(**context) -> str:
    f1 = _parse_eval_f1()
    context["ti"].xcom_push(key="eval_f1", value=f1)
    if f1 < MIN_F1:
        return "notify_failure"
    return "notify_success"


def clear_trigger_file() -> None:
    try:
        if os.path.isfile(TRIGGER_FILE):
            os.remove(TRIGGER_FILE)
    except OSError:
        pass


def notify_backend(status: str, message: str) -> None:
    import urllib.parse
    import urllib.request

    qs = urllib.parse.urlencode({"status": status, "message": message})
    req = urllib.request.Request(
        f"{BACKEND_URL}/retrain/complete?{qs}",
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception:
        pass


def on_success(**context) -> None:
    f1 = context["ti"].xcom_pull(task_ids="quality_gate", key="eval_f1") or _parse_eval_f1()
    clear_trigger_file()
    notify_backend("success", f"Retrain complete; macro F1={f1:.4f}")


def on_failure(**context) -> None:
    f1 = context["ti"].xcom_pull(task_ids="quality_gate", key="eval_f1") or _parse_eval_f1()
    clear_trigger_file()
    notify_backend("failed", f"Quality gate failed; macro F1={f1:.4f} < {MIN_F1}")


_schedule = os.environ.get("RETRAIN_SCHEDULE", "").strip()
if _schedule.lower() in {"", "none", "null"}:
    _schedule_interval = None
else:
    _schedule_interval = _schedule

with DAG(
    dag_id="saint_retrain",
    default_args=default_args,
    description="LSTM retraining with MLflow staging and F1 quality gate",
    schedule_interval=_schedule_interval,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["saint-ops", "mlops"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=f"test -f {REPO_ROOT}/data/f12_clean.csv && test -f {ARTIFACTS}/feature_meta.json",
    )

    ml_pythonpath = os.environ.get("SAINT_ML_PYTHONPATH", "/opt/saint-ml/site-packages")
    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {REPO_ROOT} && python ml/train_model.py && python ml/test_model.py",
        env={
            **os.environ,
            "PYTHONPATH": f"{ml_pythonpath}:{REPO_ROOT}/ml",
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            "MLFLOW_AUTO_STAGE": "1",
            "MLFLOW_AUTO_PROMOTE": "0",
            "SAINT_MAX_EPOCHS": os.environ.get("SAINT_MAX_EPOCHS", "10"),
        },
    )

    quality_gate_task = BranchPythonOperator(
        task_id="quality_gate",
        python_callable=quality_gate,
    )

    notify_success = PythonOperator(task_id="notify_success", python_callable=on_success)
    notify_failure = PythonOperator(task_id="notify_failure", python_callable=on_failure)

    validate_data >> train_model >> quality_gate_task
    quality_gate_task >> [notify_success, notify_failure]
