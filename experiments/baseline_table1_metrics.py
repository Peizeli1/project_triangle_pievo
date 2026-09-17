#!/usr/bin/env python3
"""Reduce the Qwen Table 1 baseline track and the retained PiEvo/RADP cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

TASKS = ("MBO", "NHO", "SPO", "TMC")
BASELINE_METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
ALL_METHODS = BASELINE_METHODS + ("pievo", "radp")
SEEDS = (0, 1, 2)
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def baseline_cell(task: str, method: str, seed: int, run_dir: Path, expected_primary: int = 24) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json", {})
    manifest = load_json(run_dir / "run_manifest.json", {})
    rows = load_jsonl(run_dir / AUDIT_NAMES[task])
    evaluator_rows = [row for row in rows if str(row.get("tool_name", "")).startswith("evaluate_")]
    primary = [
        row
        for row in evaluator_rows
        if row.get("success") is True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and row.get("duplicate_suppressed") is not True
        and finite(row.get("output"))
    ]
    invalid = [row for row in evaluator_rows if row.get("success") is not True]
    duplicates = [row for row in evaluator_rows if row.get("duplicate_suppressed") is True]
    values = [float(row["output"]) for row in sorted(primary, key=lambda row: int(row.get("call_index", 0)))]
    running_best: list[float] = []
    current = -math.inf
    for value in values:
        current = max(current, value)
        running_best.append(current)
    best = max(values) if values else None
    token_rows = load_jsonl(run_dir / "llm_calls.jsonl")
    token_total = sum(int(row.get("prompt_tokens", 0) or 0) + int(row.get("completion_tokens", 0) or 0) for row in token_rows)
    manifest_identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    complete = bool(
        summary.get("status") == "completed"
        and len(values) == expected_primary
        and manifest.get("schema_version") == "baseline_table1_run_manifest.v1"
        and manifest_identity.get("task") == task
        and manifest_identity.get("method") == method
        and manifest_identity.get("seed") == seed
        and manifest_identity.get("primary_budget") == expected_primary
        and isinstance(manifest_identity.get("model"), str)
        and manifest_identity.get("model_track") == (
            "qwen3-235b-table1"
            if manifest_identity.get("model") == "qwen3-235b-a22b-instruct-2507"
            else f"{manifest_identity.get('model')}-table1"
        )
    )
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "complete": complete,
        "primary_evaluations": len(values),
        "expected_primary": expected_primary,
        "best_value": best,
        "sq_percent": best / MU_REF[task] * 100.0 if best is not None else None,
        "auoc_percent": statistics.fmean(running_best) / MU_REF[task] * 100.0 if running_best else None,
        "mu_ref": MU_REF[task],
        "invalid_attempts": len(invalid),
        "duplicate_suppressed": len(duplicates),
        "effective_invalid_rate": len(invalid) / (len(invalid) + len(values)) if invalid or values else None,
        "duplicate_rate": len(duplicates) / (len(duplicates) + len(values)) if duplicates or values else None,
        "total_evaluator_actions": len(evaluator_rows),
        "llm_calls": len(token_rows),
        "token_total": token_total,
        "apd_available": False,
        "failure": None if complete else (summary.get("error") or "missing/incomplete manifest or primary ledger"),
    }


def matched_cell(task: str, method: str, seed: int, run_dir: Path) -> dict[str, Any]:
    try:
        from experiments.official_four_task_metrics import run_metrics
    except ImportError:
        from official_four_task_metrics import run_metrics

    value = run_metrics(task, method, seed, run_dir, 24)
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "complete": bool(value.get("primary_budget_complete") and value.get("implementation_valid")),
        "primary_evaluations": value.get("primary_evaluations"),
        "expected_primary": value.get("expected_primary"),
        "best_value": value.get("best_value"),
        "sq_percent": value.get("sq_percent"),
        "auoc_percent": value.get("auoc_percent"),
        "mu_ref": value.get("mu_ref"),
        "invalid_attempts": value.get("invalid_attempts"),
        "duplicate_suppressed": value.get("duplicate_suppressed"),
        "effective_invalid_rate": value.get("invalid_rate"),
        "duplicate_rate": value.get("duplicate_rate"),
        "total_evaluator_actions": value.get("total_evaluator_calls"),
        "llm_calls": None,
        "token_total": value.get("token_total"),
        "apd_available": value.get("apd_exact_available") is True,
        "failure": None if value.get("implementation_valid") else "; ".join(value.get("implementation_invalid_reasons", [])),
    }


def aggregate_cells(cells: list[dict[str, Any]], method: str, task: str) -> dict[str, Any]:
    selected = [cell for cell in cells if cell["method"] == method and cell["task"] == task]
    complete = all(cell.get("complete") is True for cell in selected) and len(selected) == len(SEEDS)
    def values(name: str) -> list[float]:
        return [float(cell[name]) for cell in selected if finite(cell.get(name))]
    sq_values = values("sq_percent")
    auoc_values = values("auoc_percent")
    invalid_values = values("effective_invalid_rate")
    duplicate_values = values("duplicate_rate")
    return {
        "method": method,
        "task": task,
        "complete": complete,
        "seeds": [cell["seed"] for cell in selected],
        "sq_percent": stats(sq_values),
        "auoc_percent": stats(auoc_values),
        "effective_invalid_rate": stats(invalid_values),
        "duplicate_rate": stats(duplicate_values),
        "mean_primary_actions": statistics.fmean(values("total_evaluator_actions")) if values("total_evaluator_actions") else None,
        "mean_llm_calls": statistics.fmean(values("llm_calls")) if values("llm_calls") else None,
        "mean_token_total": statistics.fmean(values("token_total")) if values("token_total") else None,
        "apd_available": all(cell.get("apd_available") is True for cell in selected),
        "failures": [cell.get("failure") for cell in selected if cell.get("failure")],
    }


def markdown_table(aggregates: list[dict[str, Any]]) -> str:
    lines = [
        "| Model / Method | MBO SQ | NHO SQ | SPO SQ | TMC SQ | Average SQ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ALL_METHODS:
        row = []
        for task in TASKS:
            item = next((value for value in aggregates if value["method"] == method and value["task"] == task), None)
            if not item or not item["complete"]:
                row.append("INCOMPLETE")
            else:
                sq = item["sq_percent"]
                row.append(f"{sq['mean']:.2f} +/- {sq['std']:.2f}")
        complete_items = [value for value in aggregates if value["method"] == method and value["complete"]]
        means = [float(value["sq_percent"]["mean"]) for value in complete_items]
        average = f"{statistics.fmean(means):.2f}" if len(means) == len(TASKS) else "INCOMPLETE"
        label = {"vanilla_mas": "Vanilla MAS", "react": "ReAct", "ai_scientist_v1": "AI Scientist v1", "ai_scientist_v2": "AI Scientist v2", "ai_researcher": "AI Researcher", "piflow": "PiFlow", "pievo": "PiEvo", "radp": "RADP"}[method]
        lines.append("| " + label + " | " + " | ".join(row) + " | " + average + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-run-map", type=Path, required=True)
    parser.add_argument("--matched-run-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    args = parser.parse_args()
    baseline_map = load_json(args.baseline_run_map, {})
    matched_map = load_json(args.matched_run_map, {})
    cells: list[dict[str, Any]] = []
    for method in BASELINE_METHODS:
        for task in TASKS:
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                cells.append(baseline_cell(task, method, seed, Path(baseline_map.get(key, ""))))
    for method in ("pievo", "radp"):
        for task in TASKS:
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                cells.append(matched_cell(task, method, seed, Path(matched_map.get(key, ""))))
    aggregates = [aggregate_cells(cells, method, task) for method in ALL_METHODS for task in TASKS]
    by_task: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        task_rows = [row for row in aggregates if row["task"] == task and row["complete"]]
        radp = next((row for row in task_rows if row["method"] == "radp"), None)
        competitors = [row for row in task_rows if row["method"] != "radp"]
        by_task[task] = {
            "radp_sota": bool(radp and competitors and all(radp["sq_percent"]["mean"] > row["sq_percent"]["mean"] for row in competitors)),
            "radp_mean_sq": radp["sq_percent"]["mean"] if radp else None,
            "strongest_competitor_mean_sq": max((row["sq_percent"]["mean"] for row in competitors), default=None),
        }
    overall_values = {row["method"]: statistics.fmean([item["sq_percent"]["mean"] for item in aggregates if item["method"] == row["method"] and item["complete"]]) for row in aggregates if all(item["complete"] for item in aggregates if item["method"] == row["method"])}
    result = {
        "schema_version": "qwen_table1_reproduction.v1",
        "model": args.model,
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": 24,
        "cells": cells,
        "aggregates": aggregates,
        "sota_checks": {
            "taskwise_sq": by_task,
            "overall_mean_sq": overall_values,
            "radp_overall_sota": bool("radp" in overall_values and len(overall_values) > 1 and all(overall_values["radp"] > value for method, value in overall_values.items() if method != "radp")),
            "claim_boundary": "SOTA here means highest three-seed mean SQ in this completed matched-evaluator Table 1, not universal superiority on hidden paper assets or external benchmarks.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown_table(aggregates), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "task", "complete", "sq_mean", "sq_std", "auoc_mean", "auoc_std", "invalid_rate_mean", "duplicate_rate_mean", "primary_actions_mean", "token_total_mean"])
        for row in aggregates:
            writer.writerow([
                row["method"], row["task"], row["complete"], row["sq_percent"]["mean"], row["sq_percent"]["std"],
                row["auoc_percent"]["mean"], row["auoc_percent"]["std"], row["effective_invalid_rate"]["mean"],
                row["duplicate_rate"]["mean"], row["mean_primary_actions"], row["mean_token_total"],
            ])
    print(json.dumps({"output": str(args.output), "all_cells_complete": all(cell["complete"] for cell in cells), "sota_checks": result["sota_checks"]}, indent=2, ensure_ascii=False))
    return 0 if all(cell["complete"] for cell in cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
