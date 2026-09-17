#!/usr/bin/env python3
"""Summarize the isolated Luna limit sweep without selecting or dropping cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from experiments.formal_usage import read_cumulative_usage
except ImportError:
    from formal_usage import read_cumulative_usage


TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (3, 4, 5, 6, 7)
MU_REF = {"NHO": 2.0, "MBO": 6.5, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "NHO": "official_nho_tool_audit.jsonl",
    "MBO": "official_chembl_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def primary_rows(path: Path) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in load_jsonl(path)
            if str(row.get("tool_name", "")).startswith("evaluate_")
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and finite(row.get("output"))
        ],
        key=lambda row: int(row.get("call_index", 0)),
    )


def metrics(task: str, method: str, seed: int, run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "run_manifest.json", {})
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    rows = primary_rows(run_dir / AUDIT_NAMES[task])
    values = [float(row["output"]) for row in rows]
    running_best = []
    current = -math.inf
    for value in values:
        current = max(current, value)
        running_best.append(current)
    mu = MU_REF[task]
    usage = read_cumulative_usage(run_dir / "token_usage.json")
    total_tokens = int(usage.get("total") or 0)
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "status": manifest.get("status"),
        "model": identity.get("model"),
        "model_track": identity.get("model_track"),
        "primary_count": len(values),
        "expected_primary": 24,
        "complete": manifest.get("status") == "completed" and len(values) == 24,
        "best_value": max(values) if values else None,
        "mean_value": statistics.fmean(values) if values else None,
        "sq_percent": max(values) / mu * 100.0 if values else None,
        "auoc_percent": statistics.fmean(running_best) / mu * 100.0 if values else None,
        "invalid_attempts": sum(
            row.get("success") is not True
            for row in load_jsonl(run_dir / AUDIT_NAMES[task])
            if str(row.get("tool_name", "")).startswith("evaluate_")
        ),
        "duplicate_suppressed": sum(
            row.get("duplicate_suppressed") is True
            for row in load_jsonl(run_dir / AUDIT_NAMES[task])
        ),
        "verification_calls": sum(
            str(row.get("tool_name", "")).startswith("verify_") or row.get("verification") is True
            for row in load_jsonl(run_dir / AUDIT_NAMES[task])
        ),
        "total_tokens": total_tokens,
        "prompt_tokens": int(usage.get("prompt") or 0),
        "completion_tokens": int(usage.get("completion") or 0),
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
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/results/exploratory_limit_sweep_gpt56luna_20260812"),
    )
    args = parser.parse_args()
    plan = load_json(args.root / "limit_sweep_protocol.json", {})
    if plan.get("paper_eligible") is not False or plan.get("include_in_main_paper_tables") is not False:
        raise SystemExit("refusing to summarize a sweep without explicit paper exclusion")
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                path = args.root / f"gpt56luna_{method}_{task.lower()}_seed{seed}_formal24"
                rows.append(metrics(task, method, seed, path))
    grouped: dict[str, Any] = {}
    for task in TASKS:
        for method in METHODS:
            selected = [row for row in rows if row["task"] == task and row["method"] == method]
            grouped[f"{method}:{task}"] = {
                "task": task,
                "method": method,
                "n_cells": len(selected),
                "n_complete": sum(row["complete"] for row in selected),
                "seeds": [row["seed"] for row in selected],
                "sq_percent": mean_std([row["sq_percent"] for row in selected if finite(row["sq_percent"])]),
                "auoc_percent": mean_std([row["auoc_percent"] for row in selected if finite(row["auoc_percent"])]),
                "best_value": mean_std([row["best_value"] for row in selected if finite(row["best_value"])]),
                "total_tokens": mean_std([float(row["total_tokens"]) for row in selected if row["total_tokens"] > 0]),
            }
    paired: dict[str, Any] = {}
    for task in TASKS:
        pievo = {row["seed"]: row for row in rows if row["task"] == task and row["method"] == "pievo"}
        radp = {row["seed"]: row for row in rows if row["task"] == task and row["method"] == "radp"}
        common = sorted(set(pievo) & set(radp))
        sq_deltas = [radp[s]["sq_percent"] - pievo[s]["sq_percent"] for s in common if finite(radp[s]["sq_percent"]) and finite(pievo[s]["sq_percent"])]
        auoc_deltas = [radp[s]["auoc_percent"] - pievo[s]["auoc_percent"] for s in common if finite(radp[s]["auoc_percent"]) and finite(pievo[s]["auoc_percent"])]
        paired[task] = {
            "common_seeds": common,
            "radp_minus_pievo_sq_percent": mean_std(sq_deltas),
            "radp_minus_pievo_auoc_percent": mean_std(auoc_deltas),
            "radp_sq_wins": sum(value > 0 for value in sq_deltas),
            "radp_auoc_wins": sum(value > 0 for value in auoc_deltas),
        }
    output = {
        "schema_version": "gpt56luna_exploratory_limit_sweep_summary.v1",
        "paper_eligible": False,
        "include_in_main_paper_tables": False,
        "plan_path": str((args.root / "limit_sweep_protocol.json").resolve()),
        "plan_sha256": sha256(args.root / "limit_sweep_protocol.json"),
        "all_predeclared_cells_retained": True,
        "cell_count": len(rows),
        "complete_cell_count": sum(row["complete"] for row in rows),
        "failed_or_incomplete_cell_count": sum(not row["complete"] for row in rows),
        "cells": rows,
        "by_task_method": grouped,
        "paired_method_comparison": paired,
        "interpretation_boundary": "Use only to characterize empirical variance and failure limits; never use to choose a paper model, seed, or claim.",
    }
    path = args.root / "limit_sweep_summary.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path.resolve()), "cells": len(rows), "complete": output["complete_cell_count"], "incomplete": output["failed_or_incomplete_cell_count"]}, ensure_ascii=False))
    return 0 if output["failed_or_incomplete_cell_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
