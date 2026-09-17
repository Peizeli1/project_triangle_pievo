"""Paired significance analysis for the Qwen3-235B component study.

Reads the frozen per-cell artifact and reports, for every arm contrast:
  * paired mean difference over matched (task, seed) cells
  * exact paired sign-permutation p-value (all 2^n sign flips)
  * bootstrap 95% CI of the paired mean difference
  * per-task mean +- SEM for each arm

This replaces prose-only significance statements in the manuscript.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

CELLS = Path("experiments/results/qwen_required_suite_v1/factorial_analysis/factorial_cells.csv")
ARMS = ("pievo", "triage_only", "radp_no_gate", "radp_full")
METRICS = ("sq_percent", "auoc_budget8_carry_percent")
BOOTSTRAP = 20000
SEED = 20260820


def load() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    table: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    with CELLS.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["task"], row["arm"], int(row["seed"]))
            table[key] = {m: float(row[m]) for m in METRICS} | {
                "valid_primary": float(row["valid_primary"]),
                "complete": row["complete"] == "True",
            }
    return table


def paired(table, arm_a: str, arm_b: str, metric: str):
    """Return matched (task, seed) differences arm_a - arm_b."""
    keys = sorted({(t, s) for (t, a, s) in table if a == arm_a} &
                  {(t, s) for (t, a, s) in table if a == arm_b})
    diffs, detail = [], []
    for task, seed in keys:
        va = table[(task, arm_a, seed)][metric]
        vb = table[(task, arm_b, seed)][metric]
        diffs.append(va - vb)
        detail.append({"task": task, "seed": seed, "a": va, "b": vb, "diff": va - vb})
    return diffs, detail


def exact_sign_permutation_p(diffs: Sequence[float]) -> float:
    """Two-sided exact p-value over all 2^n sign assignments."""
    n = len(diffs)
    observed = abs(statistics.fmean(diffs))
    hits = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        flipped = statistics.fmean(s * d for s, d in zip(signs, diffs))
        if abs(flipped) >= observed - 1e-12:
            hits += 1
    return hits / total


def bootstrap_ci(diffs: Sequence[float], reps: int = BOOTSTRAP, alpha: float = 0.05):
    rng = random.Random(SEED)
    n = len(diffs)
    means = []
    for _ in range(reps):
        means.append(statistics.fmean(rng.choice(diffs) for _ in range(n)))
    means.sort()
    lo = means[int((alpha / 2) * reps)]
    hi = means[int((1 - alpha / 2) * reps) - 1]
    return lo, hi


def sem(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / (len(values) ** 0.5)


def main() -> None:
    table = load()
    out: Dict[str, object] = {
        "source": str(CELLS),
        "n_cells": len(table),
        "bootstrap_reps": BOOTSTRAP,
        "bootstrap_seed": SEED,
        "arm_summary": {},
        "contrasts": {},
        "per_task": {},
    }

    # Arm-level summary over all matched cells.
    for arm in ARMS:
        for metric in METRICS:
            vals = [v[metric] for (t, a, s), v in table.items() if a == arm]
            out["arm_summary"].setdefault(arm, {})[metric] = {
                "n": len(vals),
                "mean": statistics.fmean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "sem": sem(vals),
            }

    # Per-task mean +- SEM (what the collaborator asked to be reported).
    tasks = sorted({t for (t, a, s) in table})
    for task in tasks:
        for arm in ARMS:
            for metric in METRICS:
                vals = [table[(task, arm, s)][metric]
                        for (t, a, s) in table if t == task and a == arm]
                if not vals:
                    continue
                out["per_task"].setdefault(task, {}).setdefault(arm, {})[metric] = {
                    "n": len(vals),
                    "mean": statistics.fmean(vals),
                    "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "sem": sem(vals),
                }

    contrasts = [
        ("radp_full", "pievo"),
        ("radp_full", "radp_no_gate"),
        ("radp_full", "triage_only"),
        ("triage_only", "pievo"),
        ("radp_no_gate", "pievo"),
    ]
    for a, b in contrasts:
        for metric in METRICS:
            diffs, detail = paired(table, a, b, metric)
            if not diffs:
                continue
            lo, hi = bootstrap_ci(diffs)
            out["contrasts"].setdefault(f"{a}_minus_{b}", {})[metric] = {
                "n_pairs": len(diffs),
                "mean_diff": statistics.fmean(diffs),
                "sd_diff": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
                "sem_diff": sem(diffs),
                "exact_sign_permutation_p": exact_sign_permutation_p(diffs),
                "bootstrap_ci95": [lo, hi],
                "n_positive": sum(1 for d in diffs if d > 0),
                "cells": detail,
            }

    dest = Path("experiments/results/component_significance_20260820.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Console report.
    print("Arm summary (mean +- sem over matched task-seed cells)")
    for arm in ARMS:
        s = out["arm_summary"][arm]
        print("  %-14s SQ %6.2f +- %4.2f (n=%d)   AUOC %6.2f +- %4.2f" % (
            arm, s["sq_percent"]["mean"], s["sq_percent"]["sem"], s["sq_percent"]["n"],
            s["auoc_budget8_carry_percent"]["mean"], s["auoc_budget8_carry_percent"]["sem"]))
    print()
    print("Paired contrasts")
    for name, per_metric in out["contrasts"].items():
        for metric, r in per_metric.items():
            tag = "SQ" if metric.startswith("sq") else "AUOC"
            print("  %-26s %-5s n=%2d  diff=%+6.2f +- %4.2f  p=%.4f  CI95=[%+.2f, %+.2f]  pos=%d/%d" % (
                name, tag, r["n_pairs"], r["mean_diff"], r["sem_diff"],
                r["exact_sign_permutation_p"], r["bootstrap_ci95"][0], r["bootstrap_ci95"][1],
                r["n_positive"], r["n_pairs"]))
    print()
    print("Wrote", dest)


if __name__ == "__main__":
    main()
