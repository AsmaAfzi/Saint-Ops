#!/usr/bin/env python3
"""CronJob: verify dataset files on PVC are fresh enough."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
MAX_AGE_HOURS = float(os.environ.get("MAX_AGE_HOURS", "168"))
REQUIRED = [
    "scenario_C_annulus.csv",
    "f12_clean.csv",
]


def main() -> int:
    now = datetime.now(timezone.utc)
    stale: list[str] = []
    missing: list[str] = []

    for name in REQUIRED:
        path = DATA_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_h = (now - mtime).total_seconds() / 3600.0
        if age_h > MAX_AGE_HOURS:
            stale.append(f"{name} ({age_h:.1f}h old)")

    if missing:
        print(f"MISSING files in {DATA_DIR}: {', '.join(missing)}", file=sys.stderr)
        return 1
    if stale:
        print(f"STALE files (max {MAX_AGE_HOURS}h): {', '.join(stale)}", file=sys.stderr)
        return 1

    print(f"OK — all required datasets fresh under {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
