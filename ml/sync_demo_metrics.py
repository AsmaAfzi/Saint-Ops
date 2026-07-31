#!/usr/bin/env python3
"""
Patch the Production MLflow run with SAINT headline metrics for demo / MLOps UI.

Reads ml/results/evaluation_report.txt and logs headline-facing keys onto the
Production model version's run (macro_f1_overall, false_alarm_rate, etc.).

Usage (with stack up):
  docker compose --profile train run --rm train python ml/sync_demo_metrics.py
"""

from __future__ import annotations

import os
import sys

ML_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(ML_DIR, ".."))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from mlflow_log import apply_demo_display_metrics, parse_evaluation_report, tracking_uri
from registry import get_latest_version


def main() -> int:
    report = os.path.join(ROOT, "ml", "results", "evaluation_report.txt")
    raw = parse_evaluation_report(report)
    if not raw:
        print(f"[sync] No metrics parsed from {report}")
        return 1

    # parse_evaluation_report already applies demo mapping; re-apply on full raw
    # in case headline keys were logged separately on the run.
    metrics = apply_demo_display_metrics(raw)

    prod = get_latest_version("Production")
    if not prod:
        print("[sync] No Production model version in registry")
        return 1

    run_id = prod["run_id"]
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("[sync] mlflow not installed")
        return 1

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    for key, value in sorted(metrics.items()):
        client.log_metric(run_id, key, float(value))

    print(f"[sync] Patched Production v{prod['version']} run {run_id[:12]}…")
    print(f"       Headline F1 (macro_f1_overall) : {metrics.get('macro_f1_overall', 0):.4f}")
    print(f"       Baseline FAR (false_alarm_rate): {metrics.get('false_alarm_rate', 0) * 100:.1f}%")
    if metrics.get("supplementary_coarse_accuracy") is not None:
        print(
            f"       Coarse accuracy (supplementary): "
            f"{metrics['supplementary_coarse_accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
