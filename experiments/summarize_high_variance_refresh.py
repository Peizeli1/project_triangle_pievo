#!/usr/bin/env python3
"""Summarize only the predeclared high-variance baseline refreshes.

The reducer never rewrites a frozen table.  It reports old three-seed values
and, when all new artifacts are complete, a six-seed estimate for the exact
whitelisted cell.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
SEEDS = (0, 1, 2)
REFRESH_SEEDS = (3, 4, 5)
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
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_map(path: Path, *, cells_key: str | None = None) -> dict[str, str]:
    value = load_json(path)
    if cells_key is not None:
        value = value[cells_key]
    if "run_map" in value and isinstance(value["run_map"], dict):
        value = value["run_map"]
    if "cells" in value and isinstance(value["cells"], dict):
        value = value["cells"]
    if not isinstance(value, dict):
        raise ValueError(f"run map is not a JSON object: {path}")
    return {str(key): str(item) for key, item in value.items()}


def primary_rows(task: str, run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    path = run_dir / AUDIT_NAMES[task]
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output = row.get("output")
        if (
            str(row.get("tool_name", "")).startswith("evaluate_")
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and isinstance(output, (int, float))
            and not isinstance(output, bool)
            and math.isfinite(float(output))
        ):
            rows.append(row)
    return sorted(rows, key=lambda row: int(row.get("call_index", 0)))


def cell_metrics(task: str, method: str, seed: int, run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run_manifest.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    rows = primary_rows(task, run_dir)
    values = [float(row["output"]) for row in rows]
    running: list[float] = []
    best = -math.inf
    for value in values:
        best = max(best, value)
        running.append(best)
    evaluator_rows = []
    audit_path = run_dir / AUDIT_NAMES[task]
    if audit_path.is_file():
        evaluator_rows = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evaluator_rows = [
            row for row in evaluator_rows if str(row.get("tool_name", "")).startswith("evaluate_")
        ]
    complete = bool(
        summary.get("status") == "completed"
        and len(values) == 24
        and manifest.get("status") == "completed"
        and manifest.get("run_identity", {}).get("task") == task
        and manifest.get("run_identity", {}).get("method") == method
        and manifest.get("run_identity", {}).get("seed") == seed
        and manifest.get("run_identity", {}).get("primary_budget") == 24
    )
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "complete": complete,
        "primary_count": len(values),
        "sq_percent": best / MU_REF[task] * 100.0 if values else None,
        "auoc_percent": statistics.fmean(running) / MU_REF[task] * 100.0 if values else None,
        "invalid_attempts": sum(row.get("success") is not True for row in evaluator_rows),
        "duplicate_suppressed": sum(row.get("duplicate_suppressed") is True for row in evaluator_rows),
    }


def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [cell for cell in cells if cell.get("complete") is True]
    def stat(name: str) -> dict[str, Any]:
        values = [float(cell[name]) for cell in valid if cell.get(name) is not None]
        return {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "n": len(values),
        }
    return {
        "complete": len(valid) == len(cells),
        "seeds": [cell["seed"] for cell in valid],
        "sq_percent": stat("sq_percent"),
        "auoc_percent": stat("auoc_percent"),
        "invalid_attempts": sum(int(cell["invalid_attempts"]) for cell in valid),
        "duplicate_suppressed": sum(int(cell["duplicate_suppressed"]) for cell in valid),
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-old-map", type=Path, required=True)
    parser.add_argument("--qwen-new-root", type=Path, required=True)
    parser.add_argument("--gemini-old-index", type=Path, required=True)
    parser.add_argument("--gemini-new-root", type=Path, required=True)
    parser.add_argument("--luna-old-map", type=Path, required=True)
    parser.add_argument("--luna-new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    maps = {
        "Qwen 3 235B": (load_map(args.qwen_old_map), args.qwen_new_root, load_map(args.qwen_new_root / "baseline_run_map.json")),
        "Gemini 2.5 Flash Lite": (load_map(args.gemini_old_index, cells_key="cells"), args.gemini_new_root, load_map(args.gemini_new_root / "baseline_run_map.json")),
        "GPT-5.6-Luna": (load_map(args.luna_old_map), args.luna_new_root, load_map(args.luna_new_root / "baseline_run_map.json")),
    }
    whitelist = {
        "Qwen 3 235B": [(method, "MBO") for method in ("vanilla_mas", "ai_scientist_v1", "ai_scientist_v2", "ai_researcher", "piflow")],
        "Gemini 2.5 Flash Lite": [("vanilla_mas", "MBO"), ("react", "MBO"), ("ai_scientist_v2", "MBO"), ("ai_researcher", "MBO")],
        "GPT-5.6-Luna": [("vanilla_mas", "MBO"), ("react", "MBO"), ("ai_scientist_v2", "MBO"), ("piflow", "NHO")],
    }
    report: dict[str, Any] = {
        "schema_version": "high_variance_refresh_summary.v1",
        "old_seeds": list(SEEDS),
        "new_seeds": list(REFRESH_SEEDS),
        "primary_budget_per_cell": 24,
        "selection_rule": "existing Table 1 standard deviation >= 25 percentage points",
        "models": {},
    }
    all_complete = True
    for model, (old_map, new_root, new_map) in maps.items():
        model_rows: list[dict[str, Any]] = []
        for method, task in whitelist[model]:
            old_cells = [
                cell_metrics(
                    task,
                    method,
                    seed,
                    Path(
                        old_map.get(f"{method}:{task}:{seed}", "")
                        if Path(old_map.get(f"{method}:{task}:{seed}", "")).is_dir()
                        else (
                            ROOT
                            / "experiments/results/radp_vs_ai_researcher_qwen235b_v12_20260808/ai_researcher"
                            / f"qwen_ai_researcher_{task.lower()}_seed{seed}_formal24"
                        )
                    ),
                )
                for seed in SEEDS
            ]
            new_cells = [
                cell_metrics(
                    task,
                    method,
                    seed,
                    Path(new_map[f"{method}:{task}:{seed}"]),
                )
                for seed in REFRESH_SEEDS
            ]
            old = aggregate(old_cells)
            refreshed = aggregate(old_cells + new_cells)
            complete = refreshed["complete"]
            all_complete = all_complete and complete
            model_rows.append(
                {
                    "method": method,
                    "label": LABELS[method],
                    "task": task,
                    "old": old,
                    "refreshed": refreshed,
                    "new_cells": new_cells,
                    "refresh_applied_eligible": complete,
                }
            )
        report["models"][model] = {"whitelist": model_rows}
    report["all_whitelisted_cells_complete"] = all_complete
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# High-variance refresh summary",
        "",
        "Only whitelisted cells are shown. Frozen cells are not recomputed or replaced.",
        "",
        "| Model | Cell | Old n=3 SQ | Refreshed n=6 SQ | Old n=3 AUOC | Refreshed n=6 AUOC | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for model, value in report["models"].items():
        for row in value["whitelist"]:
            old = row["old"]
            refreshed = row["refreshed"]
            def fmt(item: dict[str, Any], name: str) -> str:
                metric = item[name]
                if metric["mean"] is None:
                    return "INCOMPLETE"
                return f"{metric['mean']:.2f} +/- {metric['std']:.2f}"
            status = "eligible" if row["refresh_applied_eligible"] else "incomplete"
            lines.append(
                f"| {model} | {row['label']} / {row['task']} | {fmt(old, 'sq_percent')} | "
                f"{fmt(refreshed, 'sq_percent')} | {fmt(old, 'auoc_percent')} | "
                f"{fmt(refreshed, 'auoc_percent')} | {status} |"
            )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_whitelisted_cells_complete": all_complete}, ensure_ascii=False))
    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
