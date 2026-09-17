#!/usr/bin/env python3
"""Recompute APD only for the whitelisted six-seed refresh cells."""

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


TASKS = ("MBO", "NHO", "SPO", "TMC")
SEEDS = (0, 1, 2)
NEW_SEEDS = (3, 4, 5)
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}
WHITELIST = {
    "Qwen 3 235B": [("vanilla_mas", "MBO"), ("ai_scientist_v1", "MBO"), ("ai_scientist_v2", "MBO"), ("ai_researcher", "MBO"), ("piflow", "MBO")],
    "Gemini 2.5 Flash Lite": [("vanilla_mas", "MBO"), ("react", "MBO"), ("ai_scientist_v2", "MBO"), ("ai_researcher", "MBO")],
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
    return {str(k): str(v) for k, v in value.items()}


def inputs(task: str, run_dir: Path) -> list[str]:
    rows = []
    for line in (run_dir / AUDIT_NAMES[task]).read_text(encoding="utf-8").splitlines():
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
    if [int(row.get("call_index", -1)) for row in rows] != list(range(1, 25)):
        raise ValueError(f"incomplete APD input stream: {run_dir}")
    return [str(row["input"]) for row in rows]


def stat(values: list[float]) -> dict[str, Any]:
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values), "n": len(values)}


def old_apd_row(path: Path, method: str, task: str) -> dict[str, Any]:
    report = load_json(path)
    for row in report.get("aggregates", []):
        if row.get("method") == method and row.get("task") == task:
            return row["apd_percent"]
    raise KeyError(f"missing old APD row: {method}:{task}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-old-map", type=Path, required=True)
    parser.add_argument("--qwen-old-apd", type=Path, required=True)
    parser.add_argument("--qwen-new-root", type=Path, required=True)
    parser.add_argument("--gemini-old-index", type=Path, required=True)
    parser.add_argument("--gemini-old-apd", type=Path, required=True)
    parser.add_argument("--gemini-new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    qwen_old = load_map(args.qwen_old_map)
    gemini_old = load_map(args.gemini_old_index, "cells")
    qwen_new = load_map(args.qwen_new_root / "baseline_run_map.json")
    gemini_new = load_map(args.gemini_new_root / "baseline_run_map.json")
    specifications = [
        ("Qwen 3 235B", qwen_old, qwen_new, args.qwen_old_apd, "qwen"),
        ("Gemini 2.5 Flash Lite", gemini_old, gemini_new, args.gemini_old_apd, "gemini"),
    ]
    all_inputs: set[str] = set()
    streams: dict[str, list[str]] = {}
    for model, old_map, new_map, _, _ in specifications:
        for method, task in WHITELIST[model]:
            for seed in (*SEEDS, *NEW_SEEDS):
                key = f"{method}:{task}:{seed}"
                path_text = old_map.get(key, "") if seed in SEEDS else new_map.get(key, "")
                path = Path(path_text)
                if not path.is_dir() and model == "Qwen 3 235B" and method == "ai_researcher" and seed in SEEDS:
                    path = Path("experiments/results/radp_vs_ai_researcher_qwen235b_v12_20260808/ai_researcher") / f"qwen_ai_researcher_{task.lower()}_seed{seed}_formal24"
                value = inputs(task, path)
                streams[f"{model}:{key}"] = value
                all_inputs.update(value)
    embeddings, device_name = embed_texts(sorted(all_inputs), select_device(args.device), 32)

    result: dict[str, Any] = {
        "schema_version": "high_variance_refresh_apd.v1",
        "old_seeds": list(SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "device": device_name,
        "cells": {},
    }
    for model, _, _, old_apd_path, _ in specifications:
        for method, task in WHITELIST[model]:
            values = [
                apd_percent([embeddings[item] for item in streams[f"{model}:{method}:{task}:{seed}"]])
                for seed in (*SEEDS, *NEW_SEEDS)
            ]
            result["cells"][f"{model}:{method}:{task}"] = {
                "old": old_apd_row(old_apd_path, method, task),
                "refreshed": stat(values),
                "per_seed": {str(seed): value for seed, value in zip((*SEEDS, *NEW_SEEDS), values, strict=True)},
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(result["cells"]), "device": device_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
