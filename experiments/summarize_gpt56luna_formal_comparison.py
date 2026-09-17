#!/usr/bin/env python3
"""Aggregate the frozen three-seed Luna PiEvo/RADP formal comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)
MU_REF = {"NHO": 2.0, "MBO": 6.5, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "NHO": "official_nho_tool_audit.jsonl",
    "MBO": "official_chembl_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}
SEED0_RUNS = {
    "pievo:MBO:0": "gpt56luna_pievo_mbo_seed0_formal24_recovery01",
    "pievo:NHO:0": "gpt56luna_pievo_nho_seed0_formal24_recovery03",
    "pievo:SPO:0": "gpt56luna_pievo_spo_seed0_formal24_recovery01",
    "pievo:TMC:0": "gpt56luna_pievo_tmc_seed0_formal24_recovery01",
    "radp:MBO:0": "gpt56luna_radp_mbo_seed0_formal24_recovery03",
    "radp:NHO:0": "gpt56luna_radp_nho_seed0_formal24_recovery01",
    "radp:SPO:0": "gpt56luna_radp_spo_seed0_formal24_recovery01",
    "radp:TMC:0": "gpt56luna_radp_tmc_seed0_formal24_recovery01",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def audit_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def primary_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        row for row in audit_rows(path)
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and finite(row.get("output"))
    ]
    return sorted(rows, key=lambda row: int(row.get("call_index", 0)))


def run_dir(
    root: Path,
    method: str,
    task: str,
    seed: int,
    seed12_root: Path,
    run_prefix: str | None,
) -> Path:
    if run_prefix:
        return root / f"{run_prefix}_{method}_{task.lower()}_seed{seed}_formal24"
    key = f"{method}:{task}:{seed}"
    if seed == 0:
        return root / SEED0_RUNS[key]
    return seed12_root / f"gpt56luna_{method}_{task.lower()}_seed{seed}_formal24"


def cell(
    root: Path,
    seed12_root: Path,
    method: str,
    task: str,
    seed: int,
    run_prefix: str | None,
    expected_model_track: str,
    expected_reasoning_effort: str,
) -> dict[str, Any]:
    path = run_dir(root, method, task, seed, seed12_root, run_prefix)
    manifest = load_json(path / "run_manifest.json", {})
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    rows = primary_rows(path / AUDIT_NAMES[task])
    values = [float(row["output"]) for row in rows]
    current = -math.inf
    running = []
    for value in values:
        current = max(current, value)
        running.append(current)
    mu = MU_REF[task]
    call_indices = [int(row.get("call_index", 0)) for row in rows]
    run_identity_ok = (
        identity.get("task") == task
        and identity.get("method") == method
        and identity.get("seed") == seed
        and identity.get("primary_budget") == 24
        and identity.get("model") == "gpt-5.6-luna"
        and identity.get("model_track") == expected_model_track
        and identity.get("stage") == "formal"
    )
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    generation = runtime.get("generation_config", {}) if isinstance(runtime, dict) else {}
    token_usage = load_json(path / "token_usage.json", [])
    token_total = 0
    if isinstance(token_usage, list):
        for snapshot in token_usage:
            if isinstance(snapshot, dict):
                usage = snapshot.get("usage", {})
                if isinstance(usage, dict):
                    total = 0
                    for item in usage.values():
                        if isinstance(item, dict) and finite(item.get("total")):
                            total += int(item["total"])
                    token_total = max(token_total, total)
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "run_dir": str(path.resolve()),
        "status": manifest.get("status"),
        "primary_count": len(values),
        "expected_primary": 24,
        "best_value": max(values) if values else None,
        "sq_percent": max(values) / mu * 100.0 if values else None,
        "auoc_percent": statistics.fmean(running) / mu * 100.0 if values else None,
        "invalid_attempts": sum(row.get("success") is not True for row in audit_rows(path / AUDIT_NAMES[task]) if str(row.get("tool_name", "")).startswith("evaluate_")),
        "duplicate_suppressed": sum(row.get("duplicate_suppressed") is True for row in audit_rows(path / AUDIT_NAMES[task])),
        "verification_calls": sum(str(row.get("tool_name", "")).startswith("verify_") or row.get("verification") is True for row in audit_rows(path / AUDIT_NAMES[task])),
        "token_total": token_total,
        "run_identity_ok": run_identity_ok,
        "generation_config": generation,
        "model_track": identity.get("model_track"),
        "expected_reasoning_effort": expected_reasoning_effort,
        "primary_indices_contiguous": call_indices == list(range(1, len(call_indices) + 1)),
        "complete": (
            manifest.get("status") == "completed"
            and run_identity_ok
            and len(values) == 24
            and call_indices == list(range(1, 25))
            and generation.get("model") == "gpt-5.6-luna"
            and generation.get("reasoning_effort") == expected_reasoning_effort
            and generation.get("enable_thinking") is False
        ),
    }


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--seed12-root", type=Path, default=Path("experiments/results/gpt56luna_formal_seed12_20260812"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/gpt56luna_formal_comparison_20260812.json"))
    parser.add_argument("--run-prefix", default=None, help="Use one root/prefix for all three seeds in a new track.")
    parser.add_argument("--model-track", default="gpt56luna")
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()
    rows = [
        cell(
            args.formal_root,
            args.seed12_root,
            method,
            task,
            seed,
            args.run_prefix,
            args.model_track,
            args.reasoning_effort,
        )
        for task in TASKS for method in METHODS for seed in SEEDS
    ]
    grouped: dict[str, Any] = {}
    for task in TASKS:
        for method in METHODS:
            selected = [row for row in rows if row["task"] == task and row["method"] == method]
            grouped[f"{method}:{task}"] = {
                "task": task,
                "method": method,
                "seeds": [row["seed"] for row in selected],
                "complete_cells": sum(row["complete"] for row in selected),
                "sq_percent": mean_std([row["sq_percent"] for row in selected if finite(row["sq_percent"])]),
                "auoc_percent": mean_std([row["auoc_percent"] for row in selected if finite(row["auoc_percent"])]),
                "best_value": mean_std([row["best_value"] for row in selected if finite(row["best_value"])]),
                "token_total": mean_std([float(row["token_total"]) for row in selected if row["token_total"] > 0]),
            }
    paired: dict[str, Any] = {}
    for task in TASKS:
        by_key = {(row["method"], row["seed"]): row for row in rows if row["task"] == task}
        pairs = [
            (by_key[("radp", seed)], by_key[("pievo", seed)])
            for seed in SEEDS
            if ("radp", seed) in by_key and ("pievo", seed) in by_key
        ]
        sq = [r["sq_percent"] - p["sq_percent"] for r, p in pairs if finite(r["sq_percent"]) and finite(p["sq_percent"])]
        auoc = [r["auoc_percent"] - p["auoc_percent"] for r, p in pairs if finite(r["auoc_percent"]) and finite(p["auoc_percent"])]
        paired[task] = {
            "seeds": SEEDS,
            "radp_minus_pievo_sq_percent": mean_std(sq),
            "radp_minus_pievo_auoc_percent": mean_std(auoc),
            "radp_sq_wins": sum(value > 0 for value in sq),
            "radp_auoc_wins": sum(value > 0 for value in auoc),
        }
    output = {
        "schema_version": "gpt56luna_formal_comparison_summary.v1",
        "paper_eligible": True,
        "model": "gpt-5.6-luna",
        "reasoning_effort": args.reasoning_effort,
        "model_track": args.model_track,
        "primary_budget_per_cell": 24,
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "cell_count": len(rows),
        "complete_cell_count": sum(row["complete"] for row in rows),
        "cells": rows,
        "by_task_method": grouped,
        "paired_method_comparison": paired,
        "claim_boundary": f"Matched released-evaluator three-seed comparison for GPT-5.6-Luna ({args.model_track}, reasoning_effort={args.reasoning_effort}); not universal external SOTA.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "cells": len(rows), "complete": output["complete_cell_count"]}, ensure_ascii=False))
    return 0 if output["complete_cell_count"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
