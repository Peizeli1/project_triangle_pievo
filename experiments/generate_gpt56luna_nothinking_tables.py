#!/usr/bin/env python3
"""Build Luna no-thinking Table 1/2 without changing frozen cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from experiments.compute_gemini_apd import apd_percent, embed_texts, select_device
except ModuleNotFoundError:
    from compute_gemini_apd import apd_percent, embed_texts, select_device


ROOT = Path(__file__).resolve().parents[1]
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
    "pievo": "PiEvo",
    "radp": "RADP",
}
REFRESHED = {
    ("vanilla_mas", "MBO"),
    ("react", "MBO"),
    ("ai_scientist_v2", "MBO"),
    ("piflow", "NHO"),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_map(path: Path) -> dict[str, str]:
    value = load_json(path)
    if isinstance(value, dict) and isinstance(value.get("run_map"), dict):
        value = value["run_map"]
    if not isinstance(value, dict):
        raise ValueError(f"invalid run map: {path}")
    return {str(k): str(v) for k, v in value.items()}


def primary_rows(task: str, run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / AUDIT_NAMES[task]
    rows: list[dict[str, Any]] = []
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
            and isinstance(row.get("input"), str)
            and isinstance(output, (int, float))
            and not isinstance(output, bool)
            and math.isfinite(float(output))
        ):
            rows.append(row)
    return sorted(rows, key=lambda row: int(row.get("call_index", -1)))


def metrics(task: str, run_dir: Path) -> tuple[float, float, list[str]]:
    rows = primary_rows(task, run_dir)
    if len(rows) != 24 or [int(row.get("call_index", -1)) for row in rows] != list(range(1, 25)):
        raise ValueError(f"incomplete Luna cell: {run_dir}")
    values = [float(row["output"]) for row in rows]
    running: list[float] = []
    best = -math.inf
    for value in values:
        best = max(best, value)
        running.append(best)
    return (
        best / MU_REF[task] * 100.0,
        statistics.fmean(running) / MU_REF[task] * 100.0,
        [str(row["input"]) for row in rows],
    )


def stat(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def fmt(value: dict[str, Any], *, bold: bool = False) -> str:
    text = f"{value['mean']:.2f} $\\pm$ {value['std']:.2f}"
    return f"\\textbf{{{text}}}" if bold else text


def markdown_fmt(value: dict[str, Any], *, bold: bool = False) -> str:
    text = f"{value['mean']:.2f} +/- {value['std']:.2f}"
    return f"**{text}**" if bold else text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--apd", type=Path, required=True)
    parser.add_argument("--run-map", type=Path, required=True)
    parser.add_argument("--refresh-root", type=Path, required=True)
    parser.add_argument("--refresh-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    aggregate = load_json(args.aggregate)
    old_rows = {(row["method"], row["task"]): row for row in aggregate["aggregates"]}
    old_apd = load_json(args.apd)
    old_apd_rows = {
        (row["method"], row["task"]): row
        for row in old_apd["aggregates"]
    }
    old_map = load_map(args.run_map)
    refresh_map = load_map(args.refresh_root / "baseline_run_map.json")
    refresh_report = load_json(args.refresh_summary)
    refresh_rows = {
        (row["method"], row["task"]): row
        for row in refresh_report["models"]["GPT-5.6-Luna"]["whitelist"]
    }

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    inputs_for_apd: dict[tuple[str, str], list[str]] = {}
    apd_seed_values: dict[tuple[str, str], list[float]] = {}
    for method in METHODS:
        for task in TASKS:
            key = (method, task)
            old = old_rows[key]
            old_apd_row = old_apd_rows[key]
            use_refresh = (
                key in REFRESHED
                and refresh_rows.get(key, {}).get("refresh_applied_eligible") is True
            )
            if use_refresh:
                refreshed = refresh_rows[key]["refreshed"]
                sq = refreshed["sq_percent"]
                auoc = refreshed["auoc_percent"]
                seed_dirs = [old_map[f"{method}:{task}:{seed}"] for seed in SEEDS]
                seed_dirs += [refresh_map[f"{method}:{task}:{seed}"] for seed in REFRESH_SEEDS]
                inputs: list[str] = []
                for run_dir in seed_dirs:
                    _, _, cell_inputs = metrics(task, Path(run_dir))
                    inputs.extend(cell_inputs)
                inputs_for_apd[key] = inputs
                rows[key] = {
                    "method": method,
                    "task": task,
                    "source": "six_seed_refresh",
                    "seeds": [*SEEDS, *REFRESH_SEEDS],
                    "sq_percent": sq,
                    "auoc_percent": auoc,
                }
            else:
                rows[key] = {
                    "method": method,
                    "task": task,
                    "source": "frozen_original",
                    "seeds": list(SEEDS),
                    "sq_percent": old["sq_percent"],
                    "auoc_percent": old["auoc_percent"],
                }
                apd_seed_values[key] = [
                    float(cell["apd_percent"])
                    for cell in old_apd["aggregates"]
                    if cell["method"] == method and cell["task"] == task
                    for _ in [0]
                ] if False else []
                # The aggregate row already contains the old APD statistics;
                # keep them exactly for frozen cells.
                rows[key]["apd_percent"] = old_apd_row["apd_percent"]

    if inputs_for_apd:
        embeddings, device_name = embed_texts(
            sorted({item for values in inputs_for_apd.values() for item in values}),
            select_device(args.device),
            32,
        )
        for key, inputs in inputs_for_apd.items():
            values: list[float] = []
            for start in range(0, len(inputs), 24):
                values.append(apd_percent([embeddings[item] for item in inputs[start : start + 24]]))
            rows[key]["apd_percent"] = stat(values)
        apd_device = device_name
    else:
        apd_device = None

    for key, row in rows.items():
        if "apd_percent" not in row:
            row["apd_percent"] = old_apd_rows[key]["apd_percent"]

    for method in METHODS:
        task_values = [rows[(method, task)]["sq_percent"]["mean"] for task in TASKS]
        rows[(method, "__average__")] = {"sq_percent": {"mean": statistics.fmean(task_values), "std": 0.0}}
        apd_values = [rows[(method, task)]["apd_percent"]["mean"] for task in TASKS]
        auoc_values = [rows[(method, task)]["auoc_percent"]["mean"] for task in TASKS]
        rows[(method, "__average__")]["apd_percent"] = {"mean": statistics.fmean(apd_values), "std": 0.0}
        rows[(method, "__average__")]["auoc_percent"] = {"mean": statistics.fmean(auoc_values), "std": 0.0}

    output = {
        "schema_version": "gpt56luna_nothinking_tables.v2",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "enable_thinking": False,
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "frozen_rule": "Only whitelisted high-variance baseline cells may use six-seed refresh; all other cells are copied from the retained three-seed artifacts.",
        "apd_device": apd_device,
        "cells": {f"{method}:{task}": value for (method, task), value in rows.items() if task != "__average__"},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Bold the largest mean in each displayed column, retaining the complete
    # mean +/- standard deviation cell.
    sq_best = {task: max(rows[(method, task)]["sq_percent"]["mean"] for method in METHODS) for task in TASKS}
    sq_best["__average__"] = max(rows[(method, "__average__")]["sq_percent"]["mean"] for method in METHODS)
    apd_best = {task: max(rows[(method, task)]["apd_percent"]["mean"] for method in METHODS) for task in TASKS}
    auoc_best = {task: max(rows[(method, task)]["auoc_percent"]["mean"] for method in METHODS) for task in TASKS}
    apd_best["__average__"] = max(rows[(method, "__average__")]["apd_percent"]["mean"] for method in METHODS)
    auoc_best["__average__"] = max(rows[(method, "__average__")]["auoc_percent"]["mean"] for method in METHODS)

    md: list[str] = [
        "# GPT-5.6-Luna no-thinking tables",
        "",
        "Only whitelisted high-variance baseline cells use the six-seed refresh; all other cells remain frozen.",
        "",
        "## Table 1: SQ",
        "",
        "| Model / Method | MBO SQ | NHO SQ | SPO SQ | TMC SQ | Average SQ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = [rows[(method, task)]["sq_percent"] for task in TASKS]
        avg = rows[(method, "__average__")]["sq_percent"]
        cells = [markdown_fmt(value, bold=math.isclose(value["mean"], sq_best[task], rel_tol=0.0, abs_tol=1e-9)) for task, value in zip(TASKS, values, strict=True)]
        cells.append(markdown_fmt(avg, bold=math.isclose(avg["mean"], sq_best["__average__"], rel_tol=0.0, abs_tol=1e-9)))
        md.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    md += [
        "",
        "## Table 2: APD and AUOC",
        "",
        "| Model / Method | MBO APD | MBO AUOC | NHO APD | NHO AUOC | SPO APD | SPO AUOC | TMC APD | TMC AUOC | Avg APD | Avg AUOC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells: list[str] = []
        for task in TASKS:
            apd = rows[(method, task)]["apd_percent"]
            auoc = rows[(method, task)]["auoc_percent"]
            cells.extend([
                markdown_fmt(apd, bold=math.isclose(apd["mean"], apd_best[task], rel_tol=0.0, abs_tol=1e-9)),
                markdown_fmt(auoc, bold=math.isclose(auoc["mean"], auoc_best[task], rel_tol=0.0, abs_tol=1e-9)),
            ])
        avg = rows[(method, "__average__")]
        cells.extend([
            f"**{avg['apd_percent']['mean']:.2f}**" if math.isclose(avg["apd_percent"]["mean"], apd_best["__average__"], abs_tol=1e-9) else f"{avg['apd_percent']['mean']:.2f}",
            f"**{avg['auoc_percent']['mean']:.2f}**" if math.isclose(avg["auoc_percent"]["mean"], auoc_best["__average__"], abs_tol=1e-9) else f"{avg['auoc_percent']['mean']:.2f}",
        ])
        md.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    args.output_json.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_json), "apd_device": apd_device}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
