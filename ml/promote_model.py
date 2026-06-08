"""
CLI for MLflow model lifecycle: compare, approve, promote, rollback.

Usage:
    python ml/promote_model.py compare
    python ml/promote_model.py approve
    python ml/promote_model.py promote [--force]
    python ml/promote_model.py rollback
    python ml/promote_model.py history

Docker:
    docker compose --profile train run --rm train python ml/promote_model.py compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ML_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from registry import (  # noqa: E402
    approve_staging,
    compare_staging_vs_production,
    list_version_history,
    promote_staging_to_production,
    rollback_production,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAINT MLflow registry lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("compare", help="Side-by-side Staging vs Production metrics")
    sub.add_parser("approve", help="Tag Staging version as approved")
    promote_p = sub.add_parser("promote", help="Promote Staging → Production (gated)")
    promote_p.add_argument("--force", action="store_true", help="Skip comparison gate")
    sub.add_parser("rollback", help="Revert Production to previous archived version")
    hist_p = sub.add_parser("history", help="List version history")
    hist_p.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.command == "compare":
        result = compare_staging_vs_production()
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("promotion_gate", {}).get("passed") else 1

    if args.command == "approve":
        result = approve_staging()
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    if args.command == "promote":
        result = promote_staging_to_production(force=args.force)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1

    if args.command == "rollback":
        result = rollback_production()
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    if args.command == "history":
        result = list_version_history(limit=args.limit)
        print(json.dumps(result, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
