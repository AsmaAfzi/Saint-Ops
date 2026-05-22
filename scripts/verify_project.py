"""
Quick smoke checks for SAINT-OPS integration (no full model training).

Usage (from repo root):
    python scripts/verify_project.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, os.path.join(ROOT, "backend"))

REQUIRED = [
    "data/f12_clean.csv",
    "data/scenario_C_annulus.csv",
    "artifacts/feature_meta.json",
    "artifacts/scaler.pkl",
    "artifacts/global_threshold.npy",
    "artifacts/normal_errors_per_feature.npy",
    "models/lstm_autoencoder.keras",
    "ml/inference.py",
    "backend/backend.py",
]


def main() -> int:
    errors: list[str] = []
    print("SAINT-OPS verify")
    print("=" * 40)

    for rel in REQUIRED:
        path = os.path.join(ROOT, rel)
        ok = os.path.isfile(path)
        print(f"  [{'OK' if ok else 'MISSING'}] {rel}")
        if not ok:
            errors.append(rel)

    if errors:
        print(f"\nMissing {len(errors)} file(s).")
        return 1

    import json

    with open(os.path.join(ROOT, "artifacts/feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    print(f"\n  Features: {meta['all_features']}")
    print(f"  Training: {meta.get('training_version', 'unknown')}")

    from inference import classify_windows, make_sequences  # noqa: E402

    import numpy as np

    rng = np.random.default_rng(0)
    err = rng.random((20, 5)) * 0.01
    labels, *_ = classify_windows(
        err, meta["all_features"], meta["env_indices"], meta["sensor_index"]
    )
    assert len(labels) == 20
    print(f"  Inference classify_windows: OK ({len(set(labels))} label types)")

    import backend as backend_mod  # noqa: E402

    backend_mod._stream = None
    backend_mod._load_error = None
    print("  Backend module import: OK")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
