#!/usr/bin/env python3
"""Compute APD for the complete Gemini v6 MBO recheck without API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from experiments.compute_gemini_apd import apd_percent, embed_texts, load_jsonl, select_device
except ModuleNotFoundError:
    from compute_gemini_apd import apd_percent, embed_texts, load_jsonl, select_device


METHODS = ("pievo", "radp", "piflow")
SEEDS = tuple(range(8))


def primary_inputs(run_dir: Path) -> list[str]:
    rows = load_jsonl(run_dir / "official_chembl_tool_audit.jsonl")
    primary = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and isinstance(row.get("input"), str)
        and isinstance(row.get("output"), (int, float))
        and math.isfinite(float(row["output"]))
    ]
    primary.sort(key=lambda row: int(row.get("call_index", -1)))
    if [int(row.get("call_index", -1)) for row in primary] != list(range(1, 25)):
        raise ValueError(f"expected 24 contiguous primary evaluations: {run_dir}")
    return [str(row["input"]) for row in primary]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs: dict[tuple[str, int], list[str]] = {}
    all_inputs: set[str] = set()
    for method in METHODS:
        for seed in SEEDS:
            run_dir = root / ("matched" if method in {"pievo", "radp"} else "baseline_piflow")
            run_dir /= f"gemini25flash_nothinking_{method}_mbo_seed{seed}_formal24"
            values = primary_inputs(run_dir)
            inputs[(method, seed)] = values
            all_inputs.update(values)
    embeddings, device = embed_texts(sorted(all_inputs), select_device(args.device), args.batch_size)
    cell_results: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        values = []
        for seed in SEEDS:
            candidates = inputs[(method, seed)]
            value = apd_percent([embeddings[item] for item in candidates])
            values.append(value)
            key = f"{method}:MBO:{seed}"
            cell_results[key] = {
                "method": method,
                "task": "MBO",
                "seed": seed,
                "apd_percent": value,
                "primary_hypotheses": len(candidates),
                "unique_hypotheses": len(set(candidates)),
                "input_sha256": hashlib.sha256("\0".join(candidates).encode()).hexdigest(),
            }
        aggregates[method] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
            "seed_values": values,
        }
    output = {
        "schema_version": "gemini_mbo_v6_apd.v1",
        "root": str(root),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "definition": {
            "metric": "Average Pairwise Distance",
            "formula": "100 * mean_{i<j} ||phi(h_i) - phi(h_j)||_2",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "pooling": "last-token pooling with left padding",
            "normalization": "L2 unit vector",
            "distance": "Euclidean",
            "primary_only": True,
        },
        "runtime": {"device": device, "batch_size": args.batch_size, "unique_hypotheses": len(embeddings)},
        "cells": cell_results,
        "aggregates": aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": aggregates, "device": device}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
