#!/usr/bin/env python3
"""Create an auditable Gemini no-thinking Table 1/2 report from fresh cells."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from gemini_model_config import MODEL_ALIAS, NATIVE_MODEL, REPORT_SCHEMA, TABLE1_TRACK
except ModuleNotFoundError:
    from experiments.gemini_model_config import MODEL_ALIAS, NATIVE_MODEL, REPORT_SCHEMA, TABLE1_TRACK


TASKS = ("MBO", "NHO", "SPO", "TMC")
BASELINE_METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
METHODS = BASELINE_METHODS + ("pievo", "radp")
SEEDS = (0, 1, 2)
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}
TRACK = TABLE1_TRACK
LABELS = {
    "vanilla_mas": "Vanilla MAS",
    "react": "ReAct",
    "ai_scientist_v1": "AI Scientist v1",
    "ai_scientist_v2": "AI Scientist v2",
    "ai_researcher": "AI Researcher",
    "piflow": "PiFlow",
    "pievo": "PiEvo",
    "radp": "RADP",
}


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def primary_rows(task: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and finite(row.get("output"))
    ]
    return sorted(selected, key=lambda row: int(row.get("call_index", 0)))


def trajectory(task: str, primary: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["output"]) for row in primary]
    running_best: list[float] = []
    current = -math.inf
    for value in values:
        current = max(current, value)
        running_best.append(current)
    return {
        "values": values,
        "best_value": max(values) if values else None,
        "sq_percent": max(values) / MU_REF[task] * 100.0 if values else None,
        "auoc_percent": statistics.fmean(running_best) / MU_REF[task] * 100.0 if running_best else None,
    }


def expected_indices(primary: list[dict[str, Any]]) -> bool:
    return [int(row.get("call_index", -1)) for row in primary] == list(range(1, 25))


def baseline_attestation(root: Path) -> bool:
    value = load_json(root / "gemini_no_thinking_attestation.json", {})
    return bool(
        value.get("model_alias") == MODEL_ALIAS
        and value.get("native_model") == NATIVE_MODEL
        and value.get("thinking_budget") == 0
        and value.get("all_calls_verified_no_thinking") is True
        and value.get("all_calls_verified_native_model") is True
        and value.get("all_calls_forwarded_selected_seed") is True
        and sorted(value.get("expected_seeds", [])) == list(SEEDS)
    )


def baseline_cell(task: str, method: str, seed: int, root: Path, run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json", {})
    manifest = load_json(run_dir / "run_manifest.json", {})
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    all_rows = load_jsonl(run_dir / AUDIT_NAMES[task])
    primary = primary_rows(task, all_rows)
    invalid = [row for row in all_rows if str(row.get("tool_name", "")).startswith("evaluate_") and row.get("success") is not True]
    duplicate = [row for row in all_rows if str(row.get("tool_name", "")).startswith("evaluate_") and row.get("duplicate_suppressed") is True]
    metrics = trajectory(task, primary)
    identity_ok = bool(
        identity.get("task") == task
        and identity.get("method") == method
        and identity.get("seed") == seed
        and identity.get("primary_budget") == 24
        and identity.get("model") == MODEL_ALIAS
        and identity.get("model_track") == f"{MODEL_ALIAS}-table1"
        and identity.get("stage") == "table1_formal_baseline"
    )
    complete = bool(
        summary.get("status") == "completed"
        and manifest.get("status") == "completed"
        and identity_ok
        and len(primary) == 24
        and expected_indices(primary)
        and baseline_attestation(root)
    )
    return {
        "task": task, "method": method, "seed": seed, "run_dir": str(run_dir.resolve()),
        "complete": complete, "primary_evaluations": len(primary),
        "invalid_attempts": len(invalid), "duplicate_suppressed": len(duplicate),
        "total_evaluator_actions": len([row for row in all_rows if str(row.get("tool_name", "")).startswith(("evaluate_", "verify_"))]),
        "apd_status": "not_claimable_exact_task_evaluator_unavailable",
        "failure": None if complete else "baseline manifest, trajectory, no-thinking, or seed attestation incomplete",
        **metrics,
    }


def matched_cell(task: str, method: str, seed: int, run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "run_manifest.json", {})
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    audit = manifest.get("audit", {}) if isinstance(manifest, dict) else {}
    all_rows = load_jsonl(run_dir / AUDIT_NAMES[task])
    primary = primary_rows(task, all_rows)
    invalid = [row for row in all_rows if str(row.get("tool_name", "")).startswith("evaluate_") and row.get("success") is not True]
    duplicate = [row for row in all_rows if str(row.get("tool_name", "")).startswith("evaluate_") and row.get("duplicate_suppressed") is True]
    bridge_rows = load_jsonl(run_dir / "gemini_no_thinking_bridge.jsonl")
    bridge_ok = bool(bridge_rows) and all(
        row.get("no_thinking_verified") is True
        and row.get("seed_forwarded") is True
        and row.get("seed_requested") == seed
        for row in bridge_rows
    )
    metrics = trajectory(task, primary)
    identity_ok = bool(
        identity.get("task") == task
        and identity.get("method") == method
        and identity.get("seed") == seed
        and identity.get("primary_budget") == 24
        and identity.get("model") == MODEL_ALIAS
        and identity.get("track") == TRACK
        and identity.get("stage") == "formal"
    )
    complete = bool(
        manifest.get("status") == "completed"
        and identity_ok
        and audit.get("primary_complete") is True
        and len(primary) == 24
        and expected_indices(primary)
        and runtime.get("all_bridge_calls_no_thinking") is True
        and runtime.get("all_bridge_calls_native_model") is True
        and runtime.get("all_bridge_calls_forwarded_seed") is True
        and bridge_ok
    )
    return {
        "task": task, "method": method, "seed": seed, "run_dir": str(run_dir.resolve()),
        "complete": complete, "primary_evaluations": len(primary),
        "invalid_attempts": len(invalid), "duplicate_suppressed": len(duplicate),
        "total_evaluator_actions": len([row for row in all_rows if str(row.get("tool_name", "")).startswith(("evaluate_", "verify_"))]),
        "apd_status": "not_claimable_exact_task_evaluator_unavailable",
        "failure": None if complete else "matched manifest, trajectory, no-thinking, or seed attestation incomplete",
        **metrics,
    }


def summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def aggregate(cells: list[dict[str, Any]], method: str, task: str) -> dict[str, Any]:
    selected = [cell for cell in cells if cell["method"] == method and cell["task"] == task]
    complete = len(selected) == len(SEEDS) and all(cell["complete"] for cell in selected)
    def values(key: str) -> list[float]:
        return [float(cell[key]) for cell in selected if finite(cell.get(key))]
    apd_values = values("apd_percent")
    return {
        "method": method,
        "task": task,
        "complete": complete,
        "seeds": [cell["seed"] for cell in selected],
        "sq_percent": summary(values("sq_percent")),
        "auoc_percent": summary(values("auoc_percent")),
        "apd_percent": summary(apd_values) if len(apd_values) == len(SEEDS) else None,
        "mean_evaluator_actions": statistics.fmean(values("total_evaluator_actions")) if values("total_evaluator_actions") else None,
        "failures": [cell["failure"] for cell in selected if cell.get("failure")],
        "apd_claimable": complete and len(apd_values) == len(SEEDS),
    }


def display(metric: dict[str, float | None] | None) -> str:
    if not metric or metric.get("mean") is None:
        return "INCOMPLETE"
    return f"{metric['mean']:.2f} +/- {metric['std']:.2f}"


def table1(aggregates: list[dict[str, Any]]) -> str:
    lines = [
        "| Model / Method | MBO SQ | NHO SQ | SPO SQ | TMC SQ | Average SQ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        items = [next((item for item in aggregates if item["method"] == method and item["task"] == task), None) for task in TASKS]
        scores = [item["sq_percent"]["mean"] for item in items if item and item["complete"]]
        average = f"{statistics.fmean(scores):.2f}" if len(scores) == len(TASKS) else "INCOMPLETE"
        lines.append("| " + LABELS[method] + " | " + " | ".join(display(item["sq_percent"] if item and item["complete"] else None) for item in items) + f" | {average} |")
    return "\n".join(lines)


def table2(aggregates: list[dict[str, Any]]) -> str:
    lines = [
        "| Model / Method | MBO APD | MBO AUOC | NHO APD | NHO AUOC | SPO APD | SPO AUOC | TMC APD | TMC AUOC | Avg APD | Avg AUOC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        items = [next((item for item in aggregates if item["method"] == method and item["task"] == task), None) for task in TASKS]
        auoc = [item["auoc_percent"]["mean"] for item in items if item and item["complete"]]
        apd = [item["apd_percent"]["mean"] for item in items if item and item["apd_claimable"]]
        average_auoc = f"{statistics.fmean(auoc):.2f}" if len(auoc) == len(TASKS) else "INCOMPLETE"
        average_apd = f"{statistics.fmean(apd):.2f}" if len(apd) == len(TASKS) else "N/A"
        values: list[str] = []
        for item in items:
            values.extend([
                display(item["apd_percent"] if item and item["apd_claimable"] else None),
                display(item["auoc_percent"] if item and item["complete"] else None),
            ])
        lines.append("| " + LABELS[method] + " | " + " | ".join(values) + f" | {average_apd} | {average_auoc} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formal-index",
        type=Path,
        help="Explicit map of one complete formal directory per task/method/seed.",
    )
    parser.add_argument(
        "--apd-json",
        type=Path,
        help="Audited APD reconstruction generated from the same formal-cell index.",
    )
    parser.add_argument("--baseline-dir-name", default="baselines")
    parser.add_argument("--matched-dir-name", default="pievo_radp")
    args = parser.parse_args()
    root = args.root.resolve()
    formal_index = load_json(args.formal_index.resolve(), {}) if args.formal_index else {}
    indexed_cells = formal_index.get("cells", {}) if isinstance(formal_index, dict) else {}
    baseline_roots = formal_index.get("baseline_roots", {}) if isinstance(formal_index, dict) else {}
    apd_source = load_json(args.apd_json.resolve(), {}) if args.apd_json else {}
    apd_cells = apd_source.get("cells", {}) if isinstance(apd_source, dict) else {}
    if args.apd_json and (
        apd_source.get("all_cells_complete") is not True
        or not isinstance(apd_cells, dict)
        or len(apd_cells) != len(TASKS) * len(METHODS) * len(SEEDS)
    ):
        raise SystemExit("APD source is incomplete or invalid")
    cells: list[dict[str, Any]] = []
    for task in TASKS:
        task_root = root / task.lower()
        baseline_root = (
            Path(str(baseline_roots[task])).resolve()
            if task in baseline_roots
            else task_root / args.baseline_dir_name
        )
        baseline_map = load_json(baseline_root / "baseline_run_map.json", {})
        matched_root = task_root / args.matched_dir_name
        matched_map = load_json(matched_root / "matched_run_map.json", {})
        for method in BASELINE_METHODS:
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                run_dir = Path(str(indexed_cells[key])) if key in indexed_cells else Path(baseline_map.get(key, ""))
                cells.append(baseline_cell(task, method, seed, baseline_root, run_dir))
        for method in ("pievo", "radp"):
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                run_dir = Path(str(indexed_cells[key])) if key in indexed_cells else Path(matched_map.get(key, ""))
                cells.append(matched_cell(task, method, seed, run_dir))
    if args.apd_json:
        for cell in cells:
            key = f"{cell['method']}:{cell['task']}:{cell['seed']}"
            apd_cell = apd_cells.get(key)
            if not isinstance(apd_cell, dict) or not finite(apd_cell.get("apd_percent")):
                raise SystemExit(f"missing APD for formal cell: {key}")
            if Path(str(apd_cell.get("run_dir", ""))).resolve() != Path(str(cell["run_dir"])).resolve():
                raise SystemExit(f"APD run directory mismatch: {key}")
            if apd_cell.get("primary_hypotheses") != 24:
                raise SystemExit(f"APD primary count mismatch: {key}")
            cell["apd_percent"] = float(apd_cell["apd_percent"])
            cell["apd_status"] = "calculated_from_released_embedding"
    aggregates = [aggregate(cells, method, task) for method in METHODS for task in TASKS]
    complete = all(cell["complete"] for cell in cells)
    task_leaders: dict[str, str | None] = {}
    for task in TASKS:
        options = [item for item in aggregates if item["task"] == task and item["complete"]]
        task_leaders[task] = max(options, key=lambda item: float(item["sq_percent"]["mean"]))["method"] if options else None
    radp_all_tasks = all(task_leaders[task] == "radp" for task in TASKS)
    result = {
        "schema_version": REPORT_SCHEMA,
        "model": MODEL_ALIAS,
        "native_model": NATIVE_MODEL,
        "thinking_budget": 0,
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": 24,
        "formal_index": str(args.formal_index.resolve()) if args.formal_index else None,
        "cells": cells,
        "aggregates": aggregates,
        "all_cells_complete": complete,
        "task_sq_leaders": task_leaders,
        "radp_highest_sq_on_every_task": radp_all_tasks if complete else False,
        "apd": (
            {
                "claimable": True,
                "source": str(args.apd_json.resolve()),
                "definition": apd_source.get("definition", {}),
            }
            if args.apd_json
            else {"claimable": False, "reason": "No audited APD reconstruction was supplied."}
        ),
        "claim_boundary": "Fresh Gemini no-thinking matched released-evaluator reproduction; not recovery of unpublished hidden paper assets or a universal SOTA claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_output = args.output.with_suffix(".json")
    json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Gemini-2.5-Flash-Lite No-Thinking Table 1 and Table 2",
        "",
        "Fresh matched released-evaluator runs. Each reported cell requires 24 primary evaluations, three seeds, Gemini native `thinkingBudget=0`, returned `thoughtsTokenCount=0`, and forwarded provider seed evidence.",
        "",
        "## Table 1. Solution Quality",
        "",
        "Values are SQ mean +/- std (%).",
        "",
        table1(aggregates),
        "",
        "## Table 2. Exploration and Exploitation",
        "",
        (
            "APD is the mean pairwise Euclidean distance between released-Qwen-embedding normalized primary hypotheses, expressed as a percentage; "
            "AUOC is calculated from evaluator-recorded running-best trajectories. Both are mean +/- std across three seeds."
            if args.apd_json
            else "APD is `N/A`: no audited APD reconstruction was supplied. AUOC values are calculated from the evaluator-recorded running-best trajectories and are mean +/- std (%)."
        ),
        "",
        table2(aggregates),
        "",
        "## Audit Status",
        "",
        f"- All 96 formal cells complete: `{complete}`",
        f"- Per-task SQ leaders: `{task_leaders}`",
        f"- RADP highest SQ on every task: `{radp_all_tasks if complete else 'not assessable until the full matrix completes'}`",
        "- Claim boundary: matched released evaluator only; no hidden-asset or universal SOTA claim.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "json_output": str(json_output), "all_cells_complete": complete, "task_sq_leaders": task_leaders}, ensure_ascii=False, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
