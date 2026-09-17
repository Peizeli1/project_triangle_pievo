#!/usr/bin/env python3
"""Temporary exploratory reducer for completed second-stage rechecks.

This file intentionally permits unequal sample counts.  It is for inspection
only: every complete artifact available for a whitelisted cell is included,
but incomplete cells are never treated as zeroes and no result is presented
as a final 9-seed estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from experiments.compute_gemini_apd import apd_percent, embed_texts, select_device
    from experiments.summarize_high_variance_refresh2 import (
        ALL_SEEDS,
        FIRST_REFRESH_SEEDS,
        LABELS,
        SEEDS,
        SECOND_REFRESH_SEEDS,
        cell_metrics,
        load_map,
        stats,
    )
except ModuleNotFoundError:
    from compute_gemini_apd import apd_percent, embed_texts, select_device
    from summarize_high_variance_refresh2 import (
        ALL_SEEDS,
        FIRST_REFRESH_SEEDS,
        LABELS,
        SEEDS,
        SECOND_REFRESH_SEEDS,
        cell_metrics,
        load_map,
        stats,
    )


SPECS = (
    ("Qwen 3 235B", "piflow", "MBO"),
    ("Gemini 2.5 Flash Lite", "vanilla_mas", "MBO"),
    ("Gemini 2.5 Flash Lite", "react", "MBO"),
    ("Gemini 2.5 Flash Lite", "ai_researcher", "MBO"),
    ("GPT-5.6-Luna no-thinking", "vanilla_mas", "MBO"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-old-map", type=Path, required=True)
    parser.add_argument("--qwen-first-root", type=Path, required=True)
    parser.add_argument("--qwen-second-root", type=Path, required=True)
    parser.add_argument("--gemini-old-index", type=Path, required=True)
    parser.add_argument("--gemini-first-root", type=Path, required=True)
    parser.add_argument("--gemini-second-root", type=Path, required=True)
    parser.add_argument("--gemini-retry-root", type=Path, required=True)
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
    retry_map = load_map(args.gemini_retry_root / "baseline_run_map.json")
    per_cell: dict[str, Any] = {}
    all_inputs: set[str] = set()

    for model, method, task in SPECS:
        old_map, first_map, second_map = maps[model]
        cells: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for seed in ALL_SEEDS:
            if model == "Gemini 2.5 Flash Lite" and method == "ai_researcher" and seed == 6:
                source_map = retry_map
            else:
                source_map = old_map if seed in SEEDS else first_map if seed in FIRST_REFRESH_SEEDS else second_map
            key = f"{method}:{task}:{seed}"
            if key not in source_map:
                missing.append({"seed": seed, "reason": "missing run-map entry"})
                continue
            try:
                value = cell_metrics(Path(source_map[key]))
            except (FileNotFoundError, ValueError) as exc:
                missing.append({"seed": seed, "reason": str(exc), "run_dir": source_map[key]})
                continue
            value["seed"] = seed
            cells.append(value)
            all_inputs.update(value["inputs"])

        if len(cells) < len(SEEDS):
            raise RuntimeError(f"base seeds are incomplete for {model}:{method}:{task}")
        per_cell[f"{model}:{method}:{task}"] = {
            "model": model,
            "method": method,
            "label": LABELS[method],
            "task": task,
            "completed_seeds": [row["seed"] for row in cells],
            "missing_seeds": missing,
            "temporary": True,
            "metrics": {
                "sq_percent": stats([row["sq_percent"] for row in cells]),
                "auoc_percent": stats([row["auoc_percent"] for row in cells]),
            },
            "per_seed": [
                {k: v for k, v in row.items() if k != "inputs"}
                for row in cells
            ],
            "inputs": [row["inputs"] for row in cells],
        }

    embeddings, device_name = embed_texts(sorted(all_inputs), select_device(args.device), 32)
    for row in per_cell.values():
        apd_values = [
            apd_percent([embeddings[item] for item in inputs])
            for inputs in row["inputs"]
        ]
        row["metrics"]["apd_percent"] = stats(apd_values)
        row["per_seed_apd_percent"] = {
            str(seed): value
            for seed, value in zip(row["completed_seeds"], apd_values, strict=True)
        }
        del row["inputs"]

    report = {
        "schema_version": "high_variance_refresh_partial.v1",
        "temporary": True,
        "selection_rule": "current displayed Table 1 SQ standard deviation >= 40.00 percentage points",
        "base_seeds": list(SEEDS),
        "first_refresh_seeds": list(FIRST_REFRESH_SEEDS),
        "second_refresh_seeds": list(SECOND_REFRESH_SEEDS),
        "note": "Exploratory only. Unequal n reflects complete artifacts available at reduction time; no incomplete run is treated as a zero and no favorable subset is selected.",
        "apd_device": device_name,
        "targets": list(per_cell.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Temporary partial high-variance refresh",
        "",
        "Exploratory only. Every complete available seed is included; incomplete runs are excluded and reported.",
        "",
        "| Model | Cell | Completed seeds | SQ | APD | AUOC | Missing |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["targets"]:
        metric = row["metrics"]
        def fmt(name: str) -> str:
            value = metric[name]
            return f"{value['mean']:.2f} +/- {value['std']:.2f} (n={value['n']})"
        missing = ", ".join(str(item["seed"]) for item in row["missing_seeds"]) or "none"
        lines.append(
            f"| {row['model']} | {row['label']} / {row['task']} | "
            f"{','.join(str(seed) for seed in row['completed_seeds'])} | "
            f"{fmt('sq_percent')} | {fmt('apd_percent')} | {fmt('auoc_percent')} | {missing} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "targets": len(per_cell), "device": device_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
