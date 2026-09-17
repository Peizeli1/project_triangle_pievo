#!/usr/bin/env python3
"""Compute APD for the frozen Qwen3-235B Table 1 matrix without API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from compute_gemini_apd import apd_percent, embed_texts, select_device
except ModuleNotFoundError:
    from experiments.compute_gemini_apd import apd_percent, embed_texts, select_device


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
DEFAULT_MODEL = "qwen3-235b-a22b-instruct-2507"
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object expected at {path}:{line_number}")
        rows.append(value)
    return rows


def primary_inputs(task: str, run_dir: Path) -> list[str]:
    rows = load_jsonl(run_dir / AUDIT_NAMES[task])
    primary = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and isinstance(row.get("input"), str)
        and isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
        and math.isfinite(float(row["output"]))
    ]
    primary.sort(key=lambda row: int(row.get("call_index", -1)))
    indices = [int(row.get("call_index", -1)) for row in primary]
    if indices != list(range(1, 25)):
        raise ValueError(f"expected 24 contiguous primary evaluations: {run_dir}; got {indices}")
    return [str(row["input"]) for row in primary]


def cell_key(method: str, task: str, seed: int) -> str:
    return f"{method}:{task}:{seed}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")

    run_map_path = args.run_map.resolve()
    run_map_payload = load_json(run_map_path)
    run_map = run_map_payload.get("run_map", run_map_payload)
    if not isinstance(run_map, dict):
        raise SystemExit("run map must be a JSON object or contain a run_map object")
    model = str(args.model).strip()
    expected_keys = {
        cell_key(method, task, seed)
        for task in TASKS
        for method in METHODS
        for seed in SEEDS
    }
    if set(run_map) != expected_keys:
        missing = sorted(expected_keys - set(run_map))
        extra = sorted(set(run_map) - expected_keys)
        raise SystemExit(f"Qwen final run map is not exactly 96 cells; missing={missing}, extra={extra}")

    inputs_by_cell: dict[str, list[str]] = {}
    all_inputs: set[str] = set()
    for key in sorted(expected_keys):
        method, task, seed_text = key.split(":")
        seed = int(seed_text)
        run_dir = Path(str(run_map[key])).resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"missing run directory: {key}: {run_dir}")
        manifest = load_json(run_dir / "run_manifest.json")
        identity = manifest.get("run_identity", {})
        if not (
            manifest.get("status") == "completed"
            and identity.get("task") == task
            and identity.get("method") == method
            and identity.get("seed") == seed
            and identity.get("primary_budget") == 24
            and identity.get("model") == model
        ):
            raise SystemExit(f"manifest identity mismatch: {key}: {run_dir}")
        inputs = primary_inputs(task, run_dir)
        inputs_by_cell[key] = inputs
        all_inputs.update(inputs)

    embeddings, device_name = embed_texts(
        sorted(all_inputs), select_device(args.device), args.batch_size
    )
    cell_results: dict[str, dict[str, Any]] = {}
    aggregate_results: list[dict[str, Any]] = []
    for method in METHODS:
        for task in TASKS:
            values: list[float] = []
            for seed in SEEDS:
                key = cell_key(method, task, seed)
                inputs = inputs_by_cell[key]
                value = apd_percent([embeddings[item] for item in inputs])
                values.append(value)
                cell_results[key] = {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "run_dir": str(Path(str(run_map[key])).resolve()),
                    "primary_hypotheses": len(inputs),
                    "unique_hypotheses": len(set(inputs)),
                    "input_sha256": hashlib.sha256("\0".join(inputs).encode()).hexdigest(),
                    "apd_percent": value,
                }
            aggregate_results.append(
                {
                    "task": task,
                    "method": method,
                    "seeds": list(SEEDS),
                    "apd_percent": {
                        "mean": statistics.fmean(values),
                        "std": statistics.stdev(values),
                    },
                }
            )

    output = {
        "schema_version": "formal_agent_track_apd.v1",
        "model": model,
        "all_cells_complete": True,
        "cell_count": len(cell_results),
        "primary_budget_per_cell": 24,
        "run_map": str(run_map_path),
        "run_map_sha256": hashlib.sha256(run_map_path.read_bytes()).hexdigest(),
        "definition": {
            "metric": "Average Pairwise Distance",
            "formula": "100 * mean_{i<j} ||phi(h_i) - phi(h_j)||_2",
            "hypotheses": "24 successful primary evaluator inputs per formal cell; verification and duplicate-suppressed rows excluded",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_pooling": "last-token pooling with left padding",
            "embedding_normalization": "L2 unit vector",
            "distance": "Euclidean",
            "unit": "percent",
            "source": "Released PiEvo embedding implementation plus the paper APD definition.",
        },
        "runtime": {
            "device": device_name,
            "batch_size": args.batch_size,
            "unique_hypotheses_embedded": len(embeddings),
        },
        "cells": cell_results,
        "aggregates": aggregate_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "model": model,
        "cells": len(cell_results),
        "unique_hypotheses": len(embeddings),
        "device": device_name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
