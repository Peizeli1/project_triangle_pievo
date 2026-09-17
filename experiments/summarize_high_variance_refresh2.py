#!/usr/bin/env python3
"""Reduce the predeclared second-stage high-variance rechecks.

Only five cells are allowed by high_variance_refresh2_protocol_20260814.md.
The reducer combines seeds 0--8, computes SQ/AUOC from the evaluator audit,
and recomputes APD from the nine primary candidate streams.  No other cell is
read as a replacement candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from experiments.compute_gemini_apd import apd_percent, embed_texts, select_device
except ModuleNotFoundError:
    from compute_gemini_apd import apd_percent, embed_texts, select_device


TASK = "MBO"
SEEDS = (0, 1, 2)
FIRST_REFRESH_SEEDS = (3, 4, 5)
SECOND_REFRESH_SEEDS = (6, 7, 8)
ALL_SEEDS = (*SEEDS, *FIRST_REFRESH_SEEDS, *SECOND_REFRESH_SEEDS)
MU_REF = 6.5
AUDIT_NAME = "official_chembl_tool_audit.jsonl"
TARGETS = {
    "Qwen 3 235B": ("piflow", "MBO"),
    "Gemini 2.5 Flash Lite": ("vanilla_mas", "MBO"),
    "Gemini 2.5 Flash Lite / ReAct": ("react", "MBO"),
    "Gemini 2.5 Flash Lite / AI Researcher": ("ai_researcher", "MBO"),
    "GPT-5.6-Luna no-thinking": ("vanilla_mas", "MBO"),
}
LABELS = {
    "vanilla_mas": "Vanilla MAS",
    "react": "ReAct",
    "ai_researcher": "AI Researcher",
    "piflow": "PiFlow",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_map(path: Path, key: str | None = None) -> dict[str, str]:
    value = load_json(path)
    if key is not None:
        value = value[key]
    if isinstance(value, dict) and isinstance(value.get("run_map"), dict):
        value = value["run_map"]
    if isinstance(value, dict) and isinstance(value.get("cells"), dict):
        value = value["cells"]
    if not isinstance(value, dict):
        raise ValueError(f"invalid run map: {path}")
    return {str(k): str(v) for k, v in value.items()}


def primary_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / AUDIT_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
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
    rows.sort(key=lambda row: int(row.get("call_index", -1)))
    if len(rows) != 24 or [int(row.get("call_index", -1)) for row in rows] != list(range(1, 25)):
        raise ValueError(f"incomplete primary stream: {run_dir}")
    return rows


def cell_metrics(run_dir: Path) -> dict[str, Any]:
    rows = primary_rows(run_dir)
    values = [float(row["output"]) for row in rows]
    running: list[float] = []
    best = -math.inf
    for value in values:
        best = max(best, value)
        running.append(best)
    return {
        "run_dir": str(run_dir.resolve()),
        "sq_percent": best / MU_REF * 100.0,
        "auoc_percent": statistics.fmean(running) / MU_REF * 100.0,
        "inputs": [str(row["input"]) for row in rows],
    }


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-old-map", type=Path, required=True)
    parser.add_argument("--qwen-first-root", type=Path, required=True)
    parser.add_argument("--qwen-second-root", type=Path, required=True)
    parser.add_argument("--gemini-old-index", type=Path, required=True)
    parser.add_argument("--gemini-first-root", type=Path, required=True)
    parser.add_argument("--gemini-second-root", type=Path, required=True)
    parser.add_argument("--luna-old-map", type=Path, required=True)
    parser.add_argument("--luna-first-root", type=Path, required=True)
    parser.add_argument("--luna-second-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    maps = {
        "Qwen 3 235B": (
            load_map(args.qwen_old_map),
            load_map(args.qwen_first_root / "baseline_run_map.json"),
            load_map(args.qwen_second_root / "baseline_run_map.json"),
        ),
        "Gemini 2.5 Flash Lite": (
            load_map(args.gemini_old_index, key="cells"),
            load_map(args.gemini_first_root / "baseline_run_map.json"),
            load_map(args.gemini_second_root / "baseline_run_map.json"),
        ),
        "GPT-5.6-Luna no-thinking": (
            load_map(args.luna_old_map),
            load_map(args.luna_first_root / "baseline_run_map.json"),
            load_map(args.luna_second_root / "baseline_run_map.json"),
        ),
    }
    specs = [
        ("Qwen 3 235B", "piflow", "MBO"),
        ("Gemini 2.5 Flash Lite", "vanilla_mas", "MBO"),
        ("Gemini 2.5 Flash Lite", "react", "MBO"),
        ("Gemini 2.5 Flash Lite", "ai_researcher", "MBO"),
        ("GPT-5.6-Luna no-thinking", "vanilla_mas", "MBO"),
    ]

    per_cell: dict[str, Any] = {}
    all_inputs: set[str] = set()
    for model, method, task in specs:
        old_map, first_map, second_map = maps[model]
        cell_rows: list[dict[str, Any]] = []
        for seed in ALL_SEEDS:
            source_map = old_map if seed in SEEDS else first_map if seed in FIRST_REFRESH_SEEDS else second_map
            key = f"{method}:{task}:{seed}"
            if key not in source_map:
                raise KeyError(f"missing {model} {key}")
            row = cell_metrics(Path(source_map[key]))
            row["seed"] = seed
            cell_rows.append(row)
            all_inputs.update(row["inputs"])
        values = {
            "sq_percent": stats([row["sq_percent"] for row in cell_rows]),
            "auoc_percent": stats([row["auoc_percent"] for row in cell_rows]),
        }
        per_cell[f"{model}:{method}:{task}"] = {
            "model": model,
            "method": method,
            "label": LABELS[method],
            "task": task,
            "seeds": list(ALL_SEEDS),
            "metrics": values,
            "per_seed": [
                {k: v for k, v in row.items() if k != "inputs"}
                for row in cell_rows
            ],
            "inputs": [row["inputs"] for row in cell_rows],
        }

    embeddings, device_name = embed_texts(sorted(all_inputs), select_device(args.device), 32)
    for row in per_cell.values():
        apd_values = [
            apd_percent([embeddings[item] for item in inputs])
            for inputs in row["inputs"]
        ]
        row["metrics"]["apd_percent"] = stats(apd_values)
        row["per_seed_apd_percent"] = {
            str(seed): value for seed, value in zip(ALL_SEEDS, apd_values, strict=True)
        }
        del row["inputs"]

    report = {
        "schema_version": "high_variance_refresh2.v1",
        "selection_rule": "current displayed Table 1 SQ standard deviation >= 40.00 percentage points",
        "old_seeds": list(SEEDS),
        "first_refresh_seeds": list(FIRST_REFRESH_SEEDS),
        "second_refresh_seeds": list(SECOND_REFRESH_SEEDS),
        "final_seeds": list(ALL_SEEDS),
        "primary_budget_per_cell": 24,
        "apd_device": device_name,
        "targets": list(per_cell.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Second-stage high-variance refresh summary",
        "",
        "Only the five cells in `high_variance_refresh2_protocol_20260814.md` are recomputed.",
        "All rows below combine seeds 0--8; no favorable subset is selected.",
        "",
        "| Model | Cell | SQ | APD | AUOC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["targets"]:
        metric = row["metrics"]
        def fmt(name: str) -> str:
            value = metric[name]
            return f"{value['mean']:.2f} +/- {value['std']:.2f}"
        lines.append(
            f"| {row['model']} | {row['label']} / {row['task']} | "
            f"{fmt('sq_percent')} | {fmt('apd_percent')} | {fmt('auoc_percent')} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "targets": len(per_cell), "device": device_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
