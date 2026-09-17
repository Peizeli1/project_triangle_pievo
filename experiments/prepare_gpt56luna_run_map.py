#!/usr/bin/env python3
"""Create the frozen 96-cell run map from the formal Luna aggregate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/results/gpt56luna_formal_matrix_aggregate_20260813.json"
OUTPUT = ROOT / "experiments/results/gpt56luna_formal_run_map_20260813.json"


def main() -> int:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = report.get("aggregates", [])
    run_map: dict[str, str] = {}
    for aggregate in cells:
        for cell in aggregate.get("cells", []):
            key = f"{cell['method']}:{cell['task']}:{cell['seed']}"
            run_map[key] = cell["run_dir"]
    expected = 8 * 4 * 3
    if len(run_map) != expected:
        raise SystemExit(f"expected {expected} cells, found {len(run_map)}")
    payload = {
        "schema_version": "gpt56luna_formal_run_map.v1",
        "model": report["model"],
        "reasoning_effort": report["reasoning_effort"],
        "primary_budget_per_cell": report["primary_budget_per_cell"],
        "seeds": report["seeds"],
        "cell_count": len(run_map),
        "source_aggregate": str(SOURCE.resolve()),
        "run_map": dict(sorted(run_map.items())),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "cells": len(run_map)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
