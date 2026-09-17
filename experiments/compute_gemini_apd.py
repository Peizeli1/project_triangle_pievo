#!/usr/bin/env python3
"""Compute auditable APD from the frozen Gemini formal trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from gemini_model_config import RUN_PREFIX
except ModuleNotFoundError:
    from experiments.gemini_model_config import RUN_PREFIX


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
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"missing audit: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid audit JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"audit object expected at {path}:{line_number}")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_key(method: str, task: str, seed: int) -> str:
    return f"{method}:{task}:{seed}"


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
        and math.isfinite(float(row["output"]))
    ]
    primary.sort(key=lambda row: int(row.get("call_index", -1)))
    indices = [int(row.get("call_index", -1)) for row in primary]
    if indices != list(range(1, 25)):
        raise ValueError(f"expected 24 contiguous primary evaluations: {run_dir}")
    return [str(row["input"]) for row in primary]


def select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_texts(texts: list[str], device_name: str, batch_size: int) -> tuple[dict[str, np.ndarray], str]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import torch.nn.functional as torch_functional
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, local_files_only=True)
    model.eval()
    model.to(device)
    embeddings: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt")
            encoded = {name: value.to(device) for name, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            # The released embedding implementation uses left padding and last-token pooling.
            pooled = hidden[:, -1]
            normalized = torch_functional.normalize(pooled.float(), p=2, dim=1)
            for text, vector in zip(batch, normalized.cpu().numpy(), strict=True):
                embeddings[text] = np.asarray(vector, dtype=np.float64)
    return embeddings, str(device)


def apd_percent(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        raise ValueError("APD requires at least two valid hypotheses")
    matrix = np.stack(vectors, axis=0)
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    return float(distances[np.triu_indices(len(vectors), k=1)].mean() * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--formal-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")

    report_path = args.report_json.resolve()
    index_path = args.formal_index.resolve()
    report = load_json(report_path)
    formal_index = load_json(index_path)
    if report.get("all_cells_complete") is not True:
        raise SystemExit("report is not a complete formal matrix")
    if formal_index.get("status") != "frozen_complete_formal_index":
        raise SystemExit("formal index is not frozen and complete")
    indexed_paths = formal_index.get("cells")
    cells = report.get("cells")
    if not isinstance(indexed_paths, dict) or not isinstance(cells, list):
        raise SystemExit("formal report or index schema is invalid")
    if len(cells) != len(TASKS) * len(METHODS) * len(SEEDS) or len(indexed_paths) != len(cells):
        raise SystemExit("formal matrix is incomplete")

    inputs_by_cell: dict[str, list[str]] = {}
    all_inputs: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise SystemExit("invalid cell record")
        task = cell.get("task")
        method = cell.get("method")
        seed = cell.get("seed")
        if task not in TASKS or method not in METHODS or seed not in SEEDS:
            raise SystemExit(f"invalid formal cell identity: {cell}")
        key = cell_key(str(method), str(task), int(seed))
        run_dir = Path(str(cell.get("run_dir", ""))).resolve()
        indexed_dir = Path(str(indexed_paths.get(key, ""))).resolve()
        if run_dir != indexed_dir or not run_dir.is_dir():
            raise SystemExit(f"formal index mismatch: {key}")
        if cell.get("complete") is not True or cell.get("primary_evaluations") != 24:
            raise SystemExit(f"incomplete formal cell: {key}")
        inputs = primary_inputs(str(task), run_dir)
        inputs_by_cell[key] = inputs
        all_inputs.update(inputs)

    embeddings, device_name = embed_texts(sorted(all_inputs), select_device(args.device), args.batch_size)
    cell_results: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in cells:
        task = str(cell["task"])
        method = str(cell["method"])
        seed = int(cell["seed"])
        key = cell_key(method, task, seed)
        inputs = inputs_by_cell[key]
        value = apd_percent([embeddings[item] for item in inputs])
        grouped[(method, task)].append(value)
        candidate_digest = hashlib.sha256("\0".join(inputs).encode("utf-8")).hexdigest()
        cell_results[key] = {
            "task": task,
            "method": method,
            "seed": seed,
            "run_dir": str(Path(str(cell["run_dir"])).resolve()),
            "primary_hypotheses": len(inputs),
            "unique_hypotheses": len(set(inputs)),
            "input_sha256": candidate_digest,
            "apd_percent": value,
        }

    aggregates = []
    for method in METHODS:
        for task in TASKS:
            values = grouped[(method, task)]
            if len(values) != len(SEEDS):
                raise SystemExit(f"missing APD seed: {method}:{task}")
            aggregates.append(
                {
                    "method": method,
                    "task": task,
                    "seeds": list(SEEDS),
                    "apd_percent": {
                        "mean": statistics.fmean(values),
                        "std": statistics.stdev(values),
                    },
                }
            )

    output = {
        "schema_version": f"{RUN_PREFIX}_apd.v1",
        "all_cells_complete": True,
        "report_json": str(report_path),
        "report_json_sha256": sha256(report_path),
        "formal_index": str(index_path),
        "formal_index_sha256": sha256(index_path),
        "definition": {
            "metric": "Average Pairwise Distance",
            "formula": "100 * mean_{i<j} ||phi(h_i) - phi(h_j)||_2",
            "hypotheses": "24 successful primary evaluator inputs per formal cell; verification and duplicate-suppressed rows excluded",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_pooling": "last-token pooling with left padding",
            "embedding_normalization": "L2 unit vector",
            "distance": "Euclidean",
            "unit": "percent",
            "source": "Released PiEvo embedding implementation plus the paper APD definition; aggregation reconstructed because the release has no APD report script.",
        },
        "runtime": {"device": device_name, "batch_size": args.batch_size, "unique_hypotheses_embedded": len(embeddings)},
        "cells": cell_results,
        "aggregates": aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cell_results), "unique_hypotheses": len(embeddings), "device": device_name}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
