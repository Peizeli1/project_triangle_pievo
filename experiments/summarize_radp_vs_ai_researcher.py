#!/usr/bin/env python3
"""Summarize one fresh Qwen3-235B RADP vs AI Researcher comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from baseline_table1_metrics import baseline_cell
from official_four_task_metrics import run_metrics


TASKS = ("MBO", "NHO", "SPO", "TMC")
SEEDS = (0, 1, 2)
MODEL = "qwen3-235b-a22b-instruct-2507"


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def complete(cell: dict[str, Any]) -> bool:
    return bool(cell.get("complete") or (cell.get("primary_budget_complete") and cell.get("implementation_valid")))


def radp_cell(root: Path, task: str, seed: int) -> dict[str, Any]:
    run_dir = root / "radp" / f"radp_{task.lower()}_seed{seed}_formal24"
    cell = run_metrics(task, "radp", seed, run_dir, 24)
    manifest = load_json(run_dir / "run_manifest.json")
    identity = manifest.get("run_identity", {})
    cell["run_dir"] = str(run_dir.resolve())
    cell["model"] = identity.get("model")
    return cell


def researcher_cell(root: Path, task: str, seed: int) -> dict[str, Any]:
    run_dir = root / "ai_researcher" / f"qwen_ai_researcher_{task.lower()}_seed{seed}_formal24"
    cell = baseline_cell(task, "ai_researcher", seed, run_dir, 24)
    manifest = load_json(run_dir / "run_manifest.json")
    identity = manifest.get("run_identity", {})
    cell["run_dir"] = str(run_dir.resolve())
    cell["model"] = identity.get("model")
    cell["invalid_rate"] = cell.get("effective_invalid_rate")
    cell["total_evaluator_calls"] = cell.get("total_evaluator_actions")
    return cell


def aggregate(task: str, method: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    complete_cells = all(complete(cell) for cell in cells)

    def values(name: str) -> list[float]:
        return [float(cell[name]) for cell in cells if finite(cell.get(name))]

    return {
        "task": task,
        "method": method,
        "complete": complete_cells,
        "seeds": [cell.get("seed") for cell in cells],
        "sq_percent": stats(values("sq_percent")),
        "auoc_percent": stats(values("auoc_percent")),
        "invalid_rate": stats(values("invalid_rate")),
        "duplicate_rate": stats(values("duplicate_rate")),
        "total_evaluator_actions": stats(values("total_evaluator_calls")),
        "token_total": stats(values("token_total")),
        "llm_calls": stats(values("llm_calls")),
        "apd_exact_available": False,
        "failures": [cell.get("failure") for cell in cells if cell.get("failure")],
    }


def paired(task: str, radp: list[dict[str, Any]], researcher: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for seed, r_cell, a_cell in zip(SEEDS, radp, researcher, strict=True):
        records.append(
            {
                "seed": seed,
                "sq_difference_radp_minus_ai_researcher": (
                    float(r_cell["sq_percent"]) - float(a_cell["sq_percent"])
                    if finite(r_cell.get("sq_percent")) and finite(a_cell.get("sq_percent"))
                    else None
                ),
                "auoc_difference_radp_minus_ai_researcher": (
                    float(r_cell["auoc_percent"]) - float(a_cell["auoc_percent"])
                    if finite(r_cell.get("auoc_percent")) and finite(a_cell.get("auoc_percent"))
                    else None
                ),
            }
        )
    return {
        "task": task,
        "seeds": records,
        "sq_difference": stats(
            [float(item["sq_difference_radp_minus_ai_researcher"]) for item in records if finite(item.get("sq_difference_radp_minus_ai_researcher"))]
        ),
        "auoc_difference": stats(
            [float(item["auoc_difference_radp_minus_ai_researcher"]) for item in records if finite(item.get("auoc_difference_radp_minus_ai_researcher"))]
        ),
    }


def show(metric: dict[str, float | None], suffix: str = "%") -> str:
    if metric["mean"] is None or metric["std"] is None:
        return "INCOMPLETE"
    return f"{metric['mean']:.2f} +/- {metric['std']:.2f}{suffix}"


def write_markdown(path: Path, aggregates: list[dict[str, Any]], pairings: list[dict[str, Any]]) -> None:
    by_key = {(item["method"], item["task"]): item for item in aggregates}
    lines = [
        "# RADP vs AI Researcher (Qwen3-235B)",
        "",
        "Fresh matched-evaluator comparison. Each cell uses seed 0/1/2 and 24 primary evaluations.",
        "",
        "| Task | RADP SQ | AI Researcher SQ | RADP AUOC | AI Researcher AUOC | SQ delta | AUOC delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        radp = by_key[("radp", task)]
        ai = by_key[("ai_researcher", task)]
        delta = next(item for item in pairings if item["task"] == task)
        lines.append(
            "| "
            + task
            + " | "
            + show(radp["sq_percent"])
            + " | "
            + show(ai["sq_percent"])
            + " | "
            + show(radp["auoc_percent"])
            + " | "
            + show(ai["auoc_percent"])
            + " | "
            + show(delta["sq_difference"])
            + " | "
            + show(delta["auoc_difference"])
            + " |"
        )
    lines.extend(
        [
            "",
            "APD is not reported: the exact APD evaluator is unavailable for this released-surrogate track.",
            "",
            "Claim boundary: this compares the two completed methods on the released PiEvo/PiFlow evaluators; it is not a recovery of the original hidden-asset leaderboard or an external-benchmark SOTA claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in TASKS:
        cells[("radp", task)] = [radp_cell(root, task, seed) for seed in SEEDS]
        cells[("ai_researcher", task)] = [researcher_cell(root, task, seed) for seed in SEEDS]
    aggregates = [aggregate(task, method, cells[(method, task)]) for task in TASKS for method in ("radp", "ai_researcher")]
    pairings = [paired(task, cells[("radp", task)], cells[("ai_researcher", task)]) for task in TASKS]
    all_cells = [cell for key in cells for cell in cells[key]]
    model_ok = all(cell.get("model") == MODEL for cell in all_cells)
    complete_ok = all(complete(cell) for cell in all_cells)
    result = {
        "schema_version": "radp_vs_ai_researcher_qwen235b.v1",
        "comparison": {
            "model": MODEL,
            "tasks": list(TASKS),
            "seeds": list(SEEDS),
            "primary_budget_per_cell": 24,
            "methods": ["radp", "ai_researcher"],
            "claim_boundary": "Released PiEvo/PiFlow matched-evaluator comparison only; not an original hidden-asset or external-benchmark SOTA claim.",
        },
        "audit": {
            "all_models_match": model_ok,
            "all_cells_complete": complete_ok,
            "apd_exact_available": False,
        },
        "cells": [cell for key in cells for cell in cells[key]],
        "aggregates": aggregates,
        "paired_differences": pairings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), aggregates, pairings)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "task", "complete", "sq_mean", "sq_std", "auoc_mean", "auoc_std", "invalid_mean", "duplicate_mean", "actions_mean", "tokens_mean"])
        for row in aggregates:
            writer.writerow([
                row["method"], row["task"], row["complete"], row["sq_percent"]["mean"], row["sq_percent"]["std"],
                row["auoc_percent"]["mean"], row["auoc_percent"]["std"], row["invalid_rate"]["mean"], row["duplicate_rate"]["mean"],
                row["total_evaluator_actions"]["mean"], row["token_total"]["mean"],
            ])
    print(json.dumps({"output": str(args.output), "all_cells_complete": complete_ok, "all_models_match": model_ok}, ensure_ascii=False))
    return 0 if complete_ok and model_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
