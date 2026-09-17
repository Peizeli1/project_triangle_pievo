"""Hyperparameter sensitivity sweep for the evidence-admission gate.

The lightweight gate realization has four free constants:

  ANOMALY_THRESHOLD      when a residual opens a diagnostic state
  REPEAT_TOLERANCE       same-channel agreement tolerance
  INDEPENDENT_TOLERANCE  cross-channel agreement tolerance
  HELDOUT_GAIN_THRESHOLD generalization margin required to authorize a revision

The manuscript previously reported only the default operating point, which
makes the constants look arbitrary. This sweep re-runs the frozen 144-trial
hidden-source benchmark while varying one constant at a time and records the
two error rates the gate is supposed to trade off:

  false-expansion rate  (revising on noise / surrogate failure / local artifact)
  missed-mechanism rate (failing to revise on a true mechanism)

It also sweeps the held-out probe *misspecification* level, i.e. the case where
the probe itself is wrong. That measures the gate's false-rejection rate when
its own admission evidence is unreliable, which the default benchmark cannot
show because held-out support is correct by construction there.

Output: experiments/results/gate_sensitivity_<date>/
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

import active_principle_discovery_benchmark as B

FALSE_EXPANSION_CONDITIONS = ("noise", "surrogate_failure", "local_artifact")
TRUE_MECHANISM_CONDITIONS = ("true_mechanism",)
SWEEP_BUDGET = 24
SWEEP_POLICY = "active_gated"

GRIDS = {
    "ANOMALY_THRESHOLD": [0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.99],
    "REPEAT_TOLERANCE": [0.02, 0.06, 0.12, 0.20, 0.30, 0.45],
    "INDEPENDENT_TOLERANCE": [0.02, 0.06, 0.12, 0.20, 0.30, 0.45],
    "HELDOUT_GAIN_THRESHOLD": [0.02, 0.08, 0.16, 0.25, 0.35, 0.44, 0.50],
}
DEFAULTS = {k: getattr(B, k) for k in GRIDS}


def run_grid(seeds: Sequence[int] = (0, 1)) -> List[B.Trial]:
    rows = []
    for seed in seeds:
        for condition in B.CONDITIONS:
            rows.append(B.run_trial(seed, condition, SWEEP_POLICY, SWEEP_BUDGET))
    return rows


def rate(rows: Sequence[B.Trial], conditions: Sequence[str], field: str) -> Dict[str, float]:
    """Mean +- sem of a 0/1 outcome over the trials in `conditions`."""
    vals: List[float] = []
    for r in rows:
        if r.condition not in conditions:
            continue
        vals.append(float(getattr(r, field)))
    if not vals:
        return {"n": 0, "mean": float("nan"), "sem": float("nan")}
    mean = statistics.fmean(vals)
    sem = (statistics.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return {"n": len(vals), "mean": mean, "sem": sem}


def score_rows(rows: Sequence[B.Trial]) -> Dict[str, object]:
    costs = [r.verification_cost for r in rows]
    acc = [float(r.final_state_correct) for r in rows if r.final_state_correct is not None]
    return {
        "false_expansion_rate": rate(rows, FALSE_EXPANSION_CONDITIONS, "false_expansion"),
        "missed_mechanism_rate": rate(rows, TRUE_MECHANISM_CONDITIONS, "missed_true_expansion"),
        "verification_cost": {
            "mean": statistics.fmean(costs),
            "sem": (statistics.stdev(costs) / len(costs) ** 0.5) if len(costs) > 1 else 0.0,
        },
        "final_state_accuracy": {
            "n": len(acc),
            "mean": statistics.fmean(acc) if acc else float("nan"),
            "sem": (statistics.stdev(acc) / len(acc) ** 0.5) if len(acc) > 1 else 0.0,
        },
    }


def sweep_constants() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, grid in GRIDS.items():
        points = []
        for value in grid:
            for k, v in DEFAULTS.items():
                setattr(B, k, v)
            setattr(B, name, value)
            rows = run_grid()
            points.append({"value": value, "is_default": value == DEFAULTS[name]} | score_rows(rows))
        for k, v in DEFAULTS.items():
            setattr(B, k, v)
        out[name] = {"default": DEFAULTS[name], "points": points}
    return out


def sweep_heldout_misspecification(levels: Sequence[float]) -> Dict[str, object]:
    """Corrupt the held-out probe itself and measure gate false rejection.

    `level` shrinks the observed held-out gain toward zero. At level 1.0 the
    probe reports no generalization even when the mechanism is real, so a
    correct-by-construction gate must eventually reject a true mechanism. This
    is the gate's own failure mode and it is what the manuscript should report
    instead of only the favourable direction.
    """
    original = B.heldout_gain
    points = []
    for level in levels:
        def corrupted(candidates, condition, round_index, _level=level, _orig=original):
            return _orig(candidates, condition, round_index) * (1.0 - _level)

        B.heldout_gain = corrupted  # type: ignore[assignment]
        rows = run_grid()
        scored = score_rows(rows)
        points.append({"heldout_attenuation": level} | scored)
    B.heldout_gain = original  # type: ignore[assignment]
    return {"points": points}


def sweep_phase_map(
    thresholds: Sequence[float], attenuations: Sequence[float]
) -> Dict[str, object]:
    """2D map over (held-out gain threshold, held-out probe attenuation).

    The flat 1D sweeps show that the default benchmark has a large separation
    margin: the true mechanism produces a held-out gain of 0.45 while a local
    artifact produces 0.00, so any threshold in between behaves identically.
    That is a property of the environment, not evidence that the constant is
    well calibrated. This map instead reports where the decision boundary
    actually lies as a function of the evidence the gate receives, so the
    constant can be read as an explicit exploration/conservatism dial.
    """
    original = B.heldout_gain
    cells = []
    for atten in attenuations:
        def corrupted(candidates, condition, round_index, _a=atten, _orig=original):
            return _orig(candidates, condition, round_index) * (1.0 - _a)

        B.heldout_gain = corrupted  # type: ignore[assignment]
        for thr in thresholds:
            B.HELDOUT_GAIN_THRESHOLD = thr
            rows = run_grid()
            scored = score_rows(rows)
            cells.append({
                "heldout_gain_threshold": thr,
                "heldout_attenuation": atten,
                "effective_true_gain": B.EVENT_OFFSET * (1.0 - atten),
                "false_expansion_rate": scored["false_expansion_rate"]["mean"],
                "missed_mechanism_rate": scored["missed_mechanism_rate"]["mean"],
                "verification_cost": scored["verification_cost"]["mean"],
            })
    B.heldout_gain = original  # type: ignore[assignment]
    B.HELDOUT_GAIN_THRESHOLD = DEFAULTS["HELDOUT_GAIN_THRESHOLD"]
    return {"thresholds": list(thresholds), "attenuations": list(attenuations), "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/results/gate_sensitivity_20260820"))
    args = parser.parse_args()

    result = {
        "schema_version": "gate_sensitivity.v1",
        "benchmark": "active_principle_discovery_benchmark",
        "policy": SWEEP_POLICY,
        "budget": SWEEP_BUDGET,
        "seeds": [0, 1],
        "trials_per_grid_point": len(B.CONDITIONS) * 2,
        "defaults": DEFAULTS,
        "true_mechanism_heldout_gain": B.EVENT_OFFSET,
        "constant_sweeps": sweep_constants(),
        "heldout_misspecification": sweep_heldout_misspecification(
            [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        ),
        "phase_map": sweep_phase_map(
            thresholds=[0.02, 0.08, 0.16, 0.25, 0.35, 0.44, 0.50],
            attenuations=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / "gate_sensitivity.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for name, blk in result["constant_sweeps"].items():
        print("==", name, "(default %s)" % blk["default"])
        for p in blk["points"]:
            mark = " <- default" if p["is_default"] else ""
            print("   %-6s  false-exp %.2f+-%.2f   missed-mech %.2f+-%.2f   ver-cost %5.2f%s" % (
                p["value"], p["false_expansion_rate"]["mean"], p["false_expansion_rate"]["sem"],
                p["missed_mechanism_rate"]["mean"], p["missed_mechanism_rate"]["sem"],
                p["verification_cost"]["mean"], mark))
    print()
    print("== held-out probe misspecification (gate's own false-rejection mode)")
    for p in result["heldout_misspecification"]["points"]:
        print("   atten %.1f   false-exp %.2f   missed-mech %.2f   final-acc %.2f" % (
            p["heldout_attenuation"], p["false_expansion_rate"]["mean"],
            p["missed_mechanism_rate"]["mean"], p["final_state_accuracy"]["mean"]))
    print()
    print("== phase map: missed-mechanism rate  (rows=threshold, cols=attenuation)")
    pm = result["phase_map"]
    print("        " + "".join("a=%.1f " % a for a in pm["attenuations"]))
    for thr in pm["thresholds"]:
        line = "  %.2f  " % thr
        for a in pm["attenuations"]:
            c = next(c for c in pm["cells"]
                     if c["heldout_gain_threshold"] == thr and c["heldout_attenuation"] == a)
            line += " %.2f " % c["missed_mechanism_rate"]
        print(line)
    print()
    print("Wrote", dest)


if __name__ == "__main__":
    main()
