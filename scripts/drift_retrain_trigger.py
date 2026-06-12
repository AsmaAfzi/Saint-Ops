#!/usr/bin/env python3
"""
CLI drift retrain trigger — polls /ops/drift-summary and POSTs /retrain/trigger.

Usage (from host):
  python scripts/drift_retrain_trigger.py --backend http://localhost:8000 --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def post_trigger(backend: str, reason: str) -> dict:
    req = urllib.request.Request(
        f"{backend.rstrip('/')}/retrain/trigger",
        data=json.dumps({"reason": reason}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser(description="SAINT-OPS drift retrain trigger")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=60, help="Poll seconds")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        try:
            summary = fetch_json(f"{args.backend.rstrip('/')}/ops/drift-summary")
            windows = summary.get("consecutive_env_drift_windows", 0)
            threshold = summary.get("retrain_threshold_windows", 50)
            print(f"env drift windows={windows} threshold={threshold}")
            if summary.get("retrain_candidate"):
                result = post_trigger(args.backend, "cli_drift_sensor")
                print(f"Triggered retrain: {result}")
                return 0
        except urllib.error.URLError as exc:
            print(f"Backend unreachable: {exc}", file=sys.stderr)
            return 1
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
