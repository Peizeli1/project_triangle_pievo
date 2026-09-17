#!/usr/bin/env python3
"""Aggregate the completed GPT-5.6-Luna formal matrix without API calls.

This reducer treats the six baseline adaptations and the PiEvo/RADP cells as
separate protocol tracks, validates every cell before aggregation, and reports
task-wise rather than universal SOTA claims.  Exploratory limit artifacts are
intentionally not read by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("MBO", "NHO", "SPO", "TMC")
SEEDS = (0, 1, 2)
BASELINES = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
METHODS = BASELINES + ("pievo", "radp")
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}
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


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def primary_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        row for row in load_jsonl(path)
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and finite(row.get("output"))
    ]
    return sorted(rows, key=lambda row: int(row.get("call_index", 0)))


def token_total(path: Path) -> int:
    total = 0
    for row in load_jsonl(path):
        total += int(row.get("prompt_tokens", 0) or 0)
        total += int(row.get("completion_tokens", 0) or 0)
    return total


def trajectory(values: list[float], task: str) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    running: list[float] = []
    best = -math.inf
    for value in values:
        best = max(best, value)
        running.append(best)
    ref = MU_REF[task]
    return best / ref * 100.0, statistics.fmean(running) / ref * 100.0


def baseline_cell(
    task: str,
    method: str,
    seed: int,
    run_dir: Path,
    *,
    expected_baseline_model_track: str,
    expected_reasoning_effort: str,
) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json", {})
    manifest = load_json(run_dir / "run_manifest.json", {})
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    rows = primary_rows(run_dir / AUDIT_NAMES[task])
    values = [float(row["output"]) for row in rows]
    sq, auoc = trajectory(values, task)
    complete = bool(
        summary.get("status") == "completed"
        and summary.get("primary_count") == 24
        and len(values) == 24
        and manifest.get("status") == "completed"
        and manifest.get("run_identity", {}).get("task") == task
        and manifest.get("run_identity", {}).get("method") == method
        and manifest.get("run_identity", {}).get("seed") == seed
        and manifest.get("run_identity", {}).get("primary_budget") == 24
        and manifest.get("run_identity", {}).get("model_track") == expected_baseline_model_track
        and runtime.get("generation_config", {}).get("model") == "gpt-5.6-luna"
        and runtime.get("generation_config", {}).get("reasoning_effort") == expected_reasoning_effort
    )
    return {
        "track": "baseline_adaptation",
        "method": method,
        "label": LABELS[method],
        "task": task,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "complete": complete,
        "primary_count": len(values),
        "best_value": max(values) if values else None,
        "sq_percent": sq,
        "auoc_percent": auoc,
        "token_total": token_total(run_dir / "llm_calls.jsonl"),
        "protocol_sha256": manifest.get("protocol", {}).get("sha256"),
        "reason": None if complete else "baseline cell failed completeness or expected Luna protocol validation",
    }


def formal_cells(path: Path, *, expected_formal_model_track: str, expected_reasoning_effort: str) -> list[dict[str, Any]]:
    aggregate = load_json(path, {})
    cells = aggregate.get("cells", []) if isinstance(aggregate, dict) else []
    selected: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        method = str(cell.get("method"))
        task = str(cell.get("task"))
        if method not in {"pievo", "radp"} or task not in TASKS:
            continue
        generation = cell.get("generation_config", {})
        complete = bool(
            cell.get("complete") is True
            and cell.get("primary_count") == 24
            and cell.get("run_identity_ok") is True
            and cell.get("model_track", "gpt56luna") == expected_formal_model_track
            and generation.get("model") == "gpt-5.6-luna"
            and generation.get("reasoning_effort") == expected_reasoning_effort
        )
        selected.append(
            {
                "track": "pievo_radp_formal",
                "method": method,
                "label": LABELS[method],
                "task": task,
                "seed": cell.get("seed"),
                "run_dir": cell.get("run_dir"),
                "complete": complete,
                "primary_count": cell.get("primary_count"),
                "best_value": cell.get("best_value"),
                "sq_percent": cell.get("sq_percent"),
                "auoc_percent": cell.get("auoc_percent"),
                "token_total": cell.get("token_total"),
                "protocol_sha256": None,
                "reason": None if complete else "formal PiEvo/RADP cell failed completeness or expected Luna protocol validation",
            }
        )
    return selected


def stats(cells: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(cell[key]) for cell in cells if finite(cell.get(key))]
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "n": len(values),
    }


def aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for task in TASKS:
            selected = [cell for cell in cells if cell["method"] == method and cell["task"] == task]
            rows.append(
                {
                    "method": method,
                    "label": LABELS[method],
                    "task": task,
                    "complete": len(selected) == 3 and all(cell["complete"] for cell in selected),
                    "seeds": [cell["seed"] for cell in selected],
                    "sq_percent": stats(selected, "sq_percent"),
                    "auoc_percent": stats(selected, "auoc_percent"),
                    "token_total": stats(selected, "token_total"),
                    "cells": selected,
                }
            )
    return rows


def winner(rows: list[dict[str, Any]], task: str, metric: str) -> dict[str, Any] | None:
    selected = [
        row for row in rows
        if row["task"] == task and row["complete"] and finite(row[metric]["mean"])
    ]
    return max(selected, key=lambda row: float(row[metric]["mean"])) if selected else None


def markdown(rows: list[dict[str, Any]], overall: dict[str, Any]) -> str:
    lines = [
        "| Model / Method | MBO SQ | NHO SQ | SPO SQ | TMC SQ | Average SQ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        task_values: list[float] = []
        rendered: list[str] = []
        for task in TASKS:
            row = next(value for value in rows if value["method"] == method and value["task"] == task)
            if row["complete"]:
                mean = float(row["sq_percent"]["mean"])
                std = float(row["sq_percent"]["std"])
                task_values.append(mean)
                rendered.append(f"{mean:.2f} +/- {std:.2f}")
            else:
                rendered.append("INCOMPLETE")
        average = f"{statistics.fmean(task_values):.2f}" if len(task_values) == len(TASKS) else "INCOMPLETE"
        lines.append(f"| {LABELS[method]} | " + " | ".join(rendered) + f" | {average} |")
    lines.append("")
    lines.append("Task-wise SQ and AUOC leaders are reported in the JSON artifact; this table is a matched GPT-5.6-Luna evaluator comparison, not a universal benchmark claim.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=Path("experiments/results/gpt56luna_table1_baselines_20260812"))
    parser.add_argument("--formal-aggregate", type=Path, default=Path("experiments/results/gpt56luna_formal_comparison_20260812.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/gpt56luna_formal_matrix_aggregate_20260813.json"))
    parser.add_argument("--baseline-model-track", default="gpt-5.6-luna-table1")
    parser.add_argument("--formal-model-track", default="gpt56luna")
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()

    run_map = load_json(args.baseline_root / "baseline_run_map.json", {})
    cells: list[dict[str, Any]] = []
    for method in BASELINES:
        for task in TASKS:
            for seed in SEEDS:
                run_dir = Path(run_map.get(f"{method}:{task}:{seed}", ""))
                cells.append(
                    baseline_cell(
                        task,
                        method,
                        seed,
                        run_dir,
                        expected_baseline_model_track=args.baseline_model_track,
                        expected_reasoning_effort=args.reasoning_effort,
                    )
                )
    cells.extend(
        formal_cells(
            args.formal_aggregate,
            expected_formal_model_track=args.formal_model_track,
            expected_reasoning_effort=args.reasoning_effort,
        )
    )
    aggregates = aggregate(cells)
    task_comparison: dict[str, Any] = {}
    for task in TASKS:
        sq = winner(aggregates, task, "sq_percent")
        auoc = winner(aggregates, task, "auoc_percent")
        radp = next(row for row in aggregates if row["task"] == task and row["method"] == "radp")
        baselines = [row for row in aggregates if row["task"] == task and row["method"] in BASELINES and row["complete"]]
        best_baseline_sq = max((float(row["sq_percent"]["mean"]) for row in baselines), default=None)
        best_baseline_auoc = max((float(row["auoc_percent"]["mean"]) for row in baselines), default=None)
        task_comparison[task] = {
            "sq_winner": {"method": sq["method"], "label": sq["label"], "mean": sq["sq_percent"]["mean"]} if sq else None,
            "auoc_winner": {"method": auoc["method"], "label": auoc["label"], "mean": auoc["auoc_percent"]["mean"]} if auoc else None,
            "radp_sq_mean": radp["sq_percent"]["mean"],
            "radp_auoc_mean": radp["auoc_percent"]["mean"],
            "radp_minus_best_baseline_sq": radp["sq_percent"]["mean"] - best_baseline_sq if best_baseline_sq is not None else None,
            "radp_minus_best_baseline_auoc": radp["auoc_percent"]["mean"] - best_baseline_auoc if best_baseline_auoc is not None else None,
            "radp_taskwise_sq_leader": bool(sq and sq["method"] == "radp"),
            "radp_taskwise_auoc_leader": bool(auoc and auoc["method"] == "radp"),
        }
    overall_sq: dict[str, float] = {}
    overall_auoc: dict[str, float] = {}
    for method in METHODS:
        selected = [row for row in aggregates if row["method"] == method and row["complete"]]
        if len(selected) == len(TASKS):
            overall_sq[method] = statistics.fmean(float(row["sq_percent"]["mean"]) for row in selected)
            overall_auoc[method] = statistics.fmean(float(row["auoc_percent"]["mean"]) for row in selected)
    sq_order = sorted(overall_sq, key=overall_sq.get, reverse=True)
    auoc_order = sorted(overall_auoc, key=overall_auoc.get, reverse=True)
    overall = {
        "mean_sq_percent": overall_sq,
        "mean_auoc_percent": overall_auoc,
        "sq_rank": sq_order,
        "auoc_rank": auoc_order,
        "radp_overall_sq_leader": bool(sq_order and sq_order[0] == "radp"),
        "radp_overall_auoc_leader": bool(auoc_order and auoc_order[0] == "radp"),
    }
    complete_count = sum(cell["complete"] for cell in cells)
    result = {
        "schema_version": "gpt56luna_formal_matrix_aggregate.v1",
        "paper_eligible": True,
        "include_exploratory_limit_results": False,
        "model": "gpt-5.6-luna",
        "reasoning_effort": args.reasoning_effort,
        "formal_model_track": args.formal_model_track,
        "baseline_model_track": args.baseline_model_track,
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": 24,
        "method_order": list(METHODS),
        "cell_count": len(cells),
        "complete_cell_count": complete_count,
        "baseline_root": str(args.baseline_root.resolve()),
        "formal_aggregate_source": str(args.formal_aggregate.resolve()),
        "aggregates": aggregates,
        "task_comparison": task_comparison,
        "overall": overall,
        "claim_boundary": {
            "supported": "Task-wise and matched-protocol GPT-5.6-Luna comparisons over all retained three-seed cells.",
            "not_supported": [
                "universal external SOTA",
                "hidden original PiEvo leaderboard recovery",
                "source-attribution performance, because these cells are ordinary optimization runs",
                "including paper-ineligible limit-sweep or source-confusion artifacts in the main table",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(aggregates, overall), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "task", "complete", "sq_mean", "sq_std", "auoc_mean", "auoc_std", "token_mean", "token_std"])
        for row in aggregates:
            writer.writerow([
                row["method"], row["task"], row["complete"],
                row["sq_percent"]["mean"], row["sq_percent"]["std"],
                row["auoc_percent"]["mean"], row["auoc_percent"]["std"],
                row["token_total"]["mean"], row["token_total"]["std"],
            ])
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cell_count": len(cells),
        "complete_cell_count": complete_count,
        "task_comparison": task_comparison,
        "overall": overall,
    }, indent=2, ensure_ascii=False))
    return 0 if len(cells) == complete_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
