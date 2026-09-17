#!/usr/bin/env python3
"""Merge the frozen no-thinking Luna baseline and formal run maps.

The older preparation script targets the medium-reasoning Luna track.  This
script is deliberately separate so the APD computation cannot silently mix
reasoning modes.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "experiments/results/gpt56luna_nothinking_table1_baselines_20260813"
FORMAL_ROOT = ROOT / "experiments/results/gpt56luna_nothinking_formal_20260813"
OUTPUT = ROOT / "experiments/results/gpt56luna_nothinking_formal_run_map_20260813.json"
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
    "pievo",
    "radp",
)
SEEDS = (0, 1, 2)


def main() -> int:
    baseline_path = BASELINE_ROOT / "baseline_run_map.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or set(baseline) != {
        f"{method}:{task}:{seed}"
        for method in METHODS[:6]
        for task in TASKS
        for seed in SEEDS
    }:
        raise SystemExit("the no-thinking Luna baseline map is not exactly 72 cells")

    run_map = dict(baseline)
    for method in ("pievo", "radp"):
        for task in TASKS:
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                run_map[key] = str(
                    FORMAL_ROOT
                    / f"gpt56luna_nothinking_{method}_{task.lower()}_seed{seed}_formal24"
                )

    expected = {
        f"{method}:{task}:{seed}"
        for method in METHODS
        for task in TASKS
        for seed in SEEDS
    }
    if set(run_map) != expected:
        raise SystemExit(
            f"merged no-thinking map mismatch: missing={sorted(expected - set(run_map))}, "
            f"extra={sorted(set(run_map) - expected)}"
        )
    if any(not Path(value).is_dir() for value in run_map.values()):
        missing = [key for key, value in run_map.items() if not Path(value).is_dir()]
        raise SystemExit(f"missing run directories: {missing[:5]}")

    payload = {
        "schema_version": "gpt56luna_nothinking_formal_run_map.v1",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "enable_thinking": False,
        "primary_budget_per_cell": 24,
        "seeds": list(SEEDS),
        "cell_count": len(run_map),
        "baseline_source": str(baseline_path.resolve()),
        "formal_source": str(FORMAL_ROOT.resolve()),
        "run_map": dict(sorted(run_map.items())),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "cells": len(run_map)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
