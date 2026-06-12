#!/usr/bin/env python3
"""CronJob: daily drift / registry comparison report (Kubernetes)."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BACKEND = os.environ.get("SAINT_BACKEND_URL", "http://saint-ops-backend:8000").rstrip("/")
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/reports"))


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{BACKEND}{path}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"drift_report_{ts}.json"

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "backend": BACKEND}
    try:
        report["ready"] = _get("/ready")
        report["compare"] = _get("/models/compare")
        report["models_info"] = _get("/models/info")
        report["status"] = "ok"
    except urllib.error.URLError as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report failed: {exc}", file=sys.stderr)
        return 1

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Drift report written to {out}")
    gate = report.get("compare", {}).get("promotion_gate", {})
    if gate and not gate.get("passed"):
        print("WARNING: staging promotion gate not passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
