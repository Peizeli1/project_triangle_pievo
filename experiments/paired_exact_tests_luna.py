#!/usr/bin/env python3
"""Run low-power paired exact permutation tests for the Luna formal matrix."""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "experiments/results/gpt56luna_nothinking_formal_matrix_aggregate_20260813.json"
APD = ROOT / "experiments/results/gpt56luna_nothinking_apd_20260813.json"
OUTPUT = ROOT / "experiments/results/gpt56luna_nothinking_paired_exact_tests_20260814.json"
TASKS = ("MBO", "NHO", "SPO", "TMC")


def exact_two_sided_p(values: list[float]) -> float:
    observed = abs(statistics.fmean(values))
    signs = list(itertools.product((-1.0, 1.0), repeat=len(values)))
    exceed = sum(
        abs(statistics.fmean([sign * value for sign, value in zip(pattern, values, strict=True)]))
        >= observed - 1e-12
        for pattern in signs
    )
    return exceed / len(signs)


def cells_from(path: Path, metric: str) -> dict[tuple[str, str, int], float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if metric == "apd_percent":
        return {
            (str(row["method"]), str(row["task"]), int(row["seed"])): float(row[metric])
            for row in value["cells"].values()
        }
    result: dict[tuple[str, str, int], float] = {}
    for row in value["aggregates"]:
        for cell in row.get("cells", []):
            result[(str(cell["method"]), str(cell["task"]), int(cell["seed"]))] = float(cell[metric])
    return result


def main() -> int:
    metrics = {
        "SQ": cells_from(AGGREGATE, "sq_percent"),
        "AUOC": cells_from(AGGREGATE, "auoc_percent"),
        "APD": cells_from(APD, "apd_percent"),
    }
    rows = []
    for metric, values in metrics.items():
        for task in TASKS:
            differences = [
                values[("radp", task, seed)] - values[("pievo", task, seed)]
                for seed in (0, 1, 2)
            ]
            rows.append(
                {
                    "metric": metric,
                    "task": task,
                    "seeds": [0, 1, 2],
                    "radp_minus_pievo": differences,
                    "mean_delta": statistics.fmean(differences),
                    "exact_two_sided_p": exact_two_sided_p(differences),
                    "test": "paired sign permutation over three fixed seeds",
                    "interpretation": "descriptive only; n=3 gives coarse p-values and does not establish significance",
                }
            )
    payload = {
        "schema_version": "paired_exact_tests.v1",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "comparand": "RADP vs PiEvo",
        "rows": rows,
        "claim_boundary": "No significance or universal SOTA claim is made from these three-seed tests.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "tests": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
