"""
SAINT — Synthetic Drift Injection Pipeline
Input:  f12_clean.csv   (output of prepare_f12_data.py)
Output: scenario_A_env_full.csv
        scenario_B_env_single.csv
        scenario_C_annulus.csv
        scenarios_combined.csv

Three scenarios, each physically grounded in real O&G failure modes:

  Scenario A — All 4 env features drift together (reservoir depletion)
  Scenario B — Only AVG_DOWNHOLE_PRESSURE drifts (transducer calibration bias)
  Scenario C — Only AVG_ANNULUS_PRESS drifts (micro-annulus leak)

CHANGE FROM v1:
    AVG_DP_TUBING removed from Scenario A injections, matching its removal
    from ENV_FEATURES in prepare_f12_data.py.
    NOISE_FLOORS updated accordingly — AVG_DP_TUBING entry removed.
    Scenario A now injects into 4 features instead of 5.
    All other injection parameters unchanged.

Usage:
    python inject_drift.py
"""

import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_FILE = "data/f12_clean.csv"

SCENARIOS = {

    "A_env_full": {
        "filename": "scenario_A_env_full.csv",
        "description": "All 4 env features drift together — reservoir depletion",
        "label": "Environmental Drift",
        "injections": {
            # DH pressure drops as reservoir depletes.
            # -15 bar = 5.5% from baseline mean. SNR = 3.4x noise floor.
            "AVG_DOWNHOLE_PRESSURE":    {"magnitude": -15.0,   "noise_scale": 2.0},

            # Temperature barely changes — geothermal gradient is unaffected.
            # -1.5°C = 1.4% from baseline mean. SNR = 16.6x (very stable feature).
            "AVG_DOWNHOLE_TEMPERATURE": {"magnitude": -1.5,    "noise_scale": 0.05},

            # Oil volume falls significantly as reservoir pressure drops.
            # -1200 m3/day = 32% from baseline mean. SNR = 2.6x.
            # (Oil volume has high daily noise — needs larger absolute magnitude.)
            "BORE_OIL_VOL":             {"magnitude": -1200.0, "noise_scale": 150.0},

            # WHP drops proportionally with DH pressure.
            # -22 bar = 26% from baseline mean. SNR = 3.0x.
            "AVG_WHP_P":                {"magnitude": -22.0,   "noise_scale": 2.5},

            # REMOVED: AVG_DP_TUBING — 0.99 corr with DH pressure, adds no
            # information, high noise causes permanent false positives.
        },
        # AVG_ANNULUS_PRESS intentionally NOT modified.
        # If SAINT flags annulus during env drift → false positive.
    },

    "B_env_single": {
        "filename": "scenario_B_env_single.csv",
        "description": "Only DH pressure drifts — pressure transducer calibration bias",
        "label": "Sensor Drift",
        "injections": {
            # Positive transducer bias: reads higher than actual reservoir pressure.
            # +25 bar = 9.7% from baseline mean. SNR = 5.6x.
            # Key signature: temperature, oil vol, WHP all stay normal —
            # physically impossible if the reservoir actually pressurised.
            "AVG_DOWNHOLE_PRESSURE":    {"magnitude": +25.0,   "noise_scale": 1.5},
        },
    },

    "C_annulus": {
        "filename": "scenario_C_annulus.csv",
        "description": "Only annulus pressure drifts — micro-annulus leak",
        "label": "Sensor Drift",
        "injections": {
            # +12 bar over 173 days = 0.069 bar/day.
            # Industry reference for micro-annulus: 0.05-0.15 bar/day ✓
            # SNR = 5.5x noise floor.
            # All 4 env features untouched.
            "AVG_ANNULUS_PRESS":        {"magnitude": +12.0,   "noise_scale": 0.8},
        },
    },
}

# CHANGED: AVG_DP_TUBING entry removed.
NOISE_FLOORS = {
    "AVG_DOWNHOLE_PRESSURE":    4.47,
    "AVG_DOWNHOLE_TEMPERATURE": 0.09,
    "BORE_OIL_VOL":             460.16,
    "AVG_WHP_P":                7.38,
    "AVG_ANNULUS_PRESS":        2.17,
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def gradual_drift_signal(n_steps: int, magnitude: float,
                         noise_scale: float, seed: int) -> np.ndarray:
    """
    Generate a realistic gradual drift signal over n_steps timesteps.

    Three components:
      1. Linear ramp    — starts at 0, reaches `magnitude` at the last step
      2. Sinusoidal osc — 15% amplitude, 3 cycles; mimics operational variability
      3. Gaussian noise — std = noise_scale; mimics sensor measurement uncertainty
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 1, n_steps)

    linear      = magnitude * t
    oscillation = abs(magnitude) * 0.15 * np.sin(2 * np.pi * 3 * t)
    noise       = rng.normal(0, noise_scale, n_steps)

    return linear + oscillation + noise


def find_drift_onset(injected_signal: np.ndarray,
                     noise_threshold: float) -> int:
    """
    Return the first index where |signal| exceeds noise_threshold
    for 5 consecutive steps. Returns -1 if never.
    """
    abs_signal = np.abs(injected_signal)
    for i in range(len(abs_signal) - 4):
        if all(abs_signal[i:i+5] >= noise_threshold):
            return i
    return -1


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SAINT Drift Injection Pipeline")
    print("=" * 60)

    print(f"\nLoading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["DATEPRD"])
    print(f"  {len(df)} rows loaded")

    drift_mask = df["WINDOW"] == "drift"
    drift_idx  = df[drift_mask].index
    n_drift    = drift_mask.sum()
    print(f"  Drift window: {n_drift} rows "
          f"({df.loc[drift_idx[0], 'DATEPRD'].date()} → "
          f"{df.loc[drift_idx[-1], 'DATEPRD'].date()})")

    for scenario_name, config in SCENARIOS.items():
        print(f"\n{'─'*60}")
        print(f"Scenario {scenario_name}: {config['description']}")
        print(f"  Label:  {config['label']}")
        print(f"  Output: {config['filename']}")

        scenario_df = df.copy()
        scenario_df["TRUE_LABEL"]        = "No Drift"
        scenario_df["DRIFT_ONSET"]       = 0
        scenario_df["INJECTED_FEATURES"] = ""

        earliest_onset_idx = len(drift_idx)

        for feat, params in config["injections"].items():
            magnitude   = params["magnitude"]
            noise_scale = params["noise_scale"]
            noise_floor = NOISE_FLOORS[feat]

            signal = gradual_drift_signal(
                n_steps=n_drift,
                magnitude=magnitude,
                noise_scale=noise_scale,
                seed=hash(scenario_name + feat) % (2**31),
            )

            scenario_df.loc[drift_idx, feat] = (
                scenario_df.loc[drift_idx, feat].values + signal
            )

            onset_local = find_drift_onset(signal, noise_threshold=noise_floor * 3)
            if onset_local >= 0:
                earliest_onset_idx = min(earliest_onset_idx, onset_local)

            print(f"  {feat}:")
            print(f"    magnitude={magnitude:+.1f}  noise_scale={noise_scale}  "
                  f"SNR={abs(magnitude)/noise_floor:.1f}x  "
                  f"onset=step {onset_local}/{n_drift}")

            scenario_df.loc[drift_idx, "INJECTED_FEATURES"] = (
                scenario_df.loc[drift_idx, "INJECTED_FEATURES"]
                .apply(lambda x: (x + "," + feat).lstrip(","))
            )

        if earliest_onset_idx < len(drift_idx):
            onset_global_idx = drift_idx[earliest_onset_idx]
            scenario_df.loc[onset_global_idx:, "TRUE_LABEL"]  = config["label"]
            scenario_df.loc[onset_global_idx:, "DRIFT_ONSET"] = 1
            n_labelled = (scenario_df["DRIFT_ONSET"] == 1).sum()
            print(f"\n  Ground truth: {n_labelled} rows labelled '{config['label']}'")
            print(f"  Onset date:   "
                  f"{scenario_df.loc[onset_global_idx, 'DATEPRD'].date()}")
        else:
            print("  WARNING: drift never exceeded 3x noise threshold")

        # Sanity check: non-injected features must be unchanged
        non_injected = [f for f in NOISE_FLOORS if f not in config["injections"]]
        max_diff = max(
            np.abs(scenario_df.loc[drift_idx, f].values -
                   df.loc[drift_idx, f].values).max()
            for f in non_injected
        )
        print(f"  Sanity check (non-injected max change): {max_diff:.8f}  "
              f"{'✓' if max_diff < 1e-6 else '✗ CHECK THIS'}")

        scenario_df.to_csv(config["filename"], index=False)
        print(f"  Saved → {config['filename']}")

    # ── COMBINED FILE ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Building combined evaluation file ...")

    pieces = []
    for scenario_name, config in SCENARIOS.items():
        s_df = pd.read_csv(config["filename"], parse_dates=["DATEPRD"])
        s_df["SCENARIO"] = scenario_name
        pieces.append(s_df)

    base_df = pieces[0][pieces[0]["WINDOW"] != "drift"].copy()
    base_df["SCENARIO"] = "all"
    drift_pieces = [p[p["WINDOW"] == "drift"] for p in pieces]

    combined = pd.concat([base_df] + drift_pieces, ignore_index=True)
    combined = combined.sort_values(["SCENARIO", "DATEPRD"]).reset_index(drop=True)
    combined.to_csv("scenarios_combined.csv", index=False)
    print(f"  Saved → scenarios_combined.csv  ({len(combined)} rows)")

    print(f"\n{'='*60}")
    print("Output files:")
    print("  scenario_A_env_full.csv   — 4 env features drift (Environmental Drift)")
    print("  scenario_B_env_single.csv — DH pressure only (Sensor Drift)")
    print("  scenario_C_annulus.csv    — annulus only (Sensor Drift)")
    print("  scenarios_combined.csv    — all 3 scenarios concatenated")
    print("\nNext step: python train_saint.py")


if __name__ == "__main__":
    main()