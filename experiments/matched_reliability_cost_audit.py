#!/usr/bin/env python3
"""Audit equal-total reliability tradeoffs on the completed matched runs.

The primary trajectories are frozen formal artifacts.  This script replays
labelled source conditions on those trajectories without making model/API
calls.  It is evidence for the reliability-cost diagnostic only; it cannot
replace an independent physical verification channel or an external
benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from official_corruption_replay_qwen_seed0 import (
    CONDITIONS,
    POLICIES,
    load_trajectory,
    run_policy,
)


TASKS = ("NHO", "MBO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)
EXPECTED_PRIMARY = 24
TOTAL_BUDGET = 48
REPLAY_POLICIES = ("source_blind", "radp_v1", "radp_v1_no_backtrack")
AUDIT_NAMES = {
    "NHO": "official_nho_tool_audit.jsonl",
    "MBO": "official_chembl_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def primary_scale(values: list[dict[str, Any]]) -> float:
    observations = [float(row["output"]) for row in values]
    scale = statistics.pstdev(observations) * 0.5
    return max(scale, 1e-6)


def run_dir(results_root: Path, method: str, task: str, seed: int) -> Path:
    return results_root / f"matched_qwen235b_{method}_{task.lower()}_seed{seed}_formal24"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["task"]),
            str(row["condition"]),
            str(row["policy"]),
            str(row["budget_mode"]),
        )
        grouped.setdefault(key, []).append(row)

    cells: dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        task, condition, policy, budget_mode = key
        def mean_field(name: str) -> float | None:
            numbers = [float(row[name]) for row in values if row.get(name) is not None]
            return statistics.mean(numbers) if numbers else None

        cell_key = ":".join(key)
        cells[cell_key] = {
            "task": task,
            "condition": condition,
            "policy": policy,
            "budget_mode": budget_mode,
            "runs": len(values),
            "false_expansion_rate": mean_field("false_expansion_rate"),
            "true_mechanism_discovery_rate": mean_field("true_mechanism_discovery_rate"),
            "backtrack_success_rate": mean_field("backtrack_success"),
            "mean_verification_cost": mean_field("verification_cost"),
            "mean_total_calls": mean_field("total_calls"),
            "budget_feasible": all(bool(row["budget_feasible"]) for row in values),
        }
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/matched_reliability_cost_audit.json"),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for method in METHODS:
        for task in TASKS:
            for seed in SEEDS:
                directory = run_dir(args.results_root, method, task, seed)
                manifest = load_json(directory / "run_manifest.json")
                if manifest.get("status") != "completed":
                    raise ValueError(f"formal run is not completed: {directory}")
                audit = directory / AUDIT_NAMES[task]
                trajectory = load_trajectory(audit, task)
                if len(trajectory) != EXPECTED_PRIMARY:
                    raise ValueError(f"{directory}: expected 24 primary rows")
                inputs.append({
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "audit": str(audit.resolve()),
                    "audit_sha256": sha256(audit),
                    "primary": len(trajectory),
                })
                scale = primary_scale(trajectory)
                for budget_mode in ("equal_primary", "equal_total"):
                    for condition in CONDITIONS:
                        for policy in REPLAY_POLICIES:
                            replay = run_policy(
                                task,
                                trajectory,
                                policy,
                                condition,
                                budget_mode,
                                scale,
                            )
                            replay.update({
                                "method": method,
                                "seed": seed,
                                "scale": scale,
                            })
                            rows.append(replay)

    cells = aggregate(rows)

    equal_total = [row for row in rows if row["budget_mode"] == "equal_total"]
    expected_rows = len(TASKS) * len(METHODS) * len(SEEDS) * len(CONDITIONS) * len(REPLAY_POLICIES)
    checks: dict[str, Any] = {
        "formal_input_runs_complete": len(inputs) == len(TASKS) * len(METHODS) * len(SEEDS),
        "each_input_has_24_primary": all(item["primary"] == EXPECTED_PRIMARY for item in inputs),
        "all_replay_cells_present": len(rows) == expected_rows * 2,
        "equal_total_rows_budget_feasible": bool(equal_total) and all(
            row["total_calls"] <= TOTAL_BUDGET and row["budget_feasible"]
            for row in equal_total
        ),
    }

    def cell(task: str, condition: str, policy: str) -> dict[str, Any]:
        return cells[":".join((task, condition, policy, "equal_total"))]

    # The reliability claim is a Pareto statement: at a fixed total-call cap,
    # RADP rejects spurious expansion and backtracks reversals, while the
    # source-blind policy remains cheaper on the clean path.
    spurious = ("one_shot_noise", "persistent_surrogate_failure", "local_artifact")
    checks["radp_rejects_spurious_equal_total"] = all(
        cell(task, condition, "radp_v1")["false_expansion_rate"] == 0.0
        for task in TASKS
        for condition in spurious
    )
    checks["source_blind_expands_spurious_equal_total"] = all(
        cell(task, condition, "source_blind")["false_expansion_rate"] > 0.0
        for task in TASKS
        for condition in spurious
    )
    checks["radp_discovers_true_mechanism_equal_total"] = all(
        cell(task, "true_mechanism", "radp_v1")["true_mechanism_discovery_rate"] == 1.0
        for task in TASKS
    )
    checks["radp_backtracks_reversal_equal_total"] = all(
        cell(task, "mechanism_reversal", "radp_v1")["backtrack_success_rate"] == 1.0
        for task in TASKS
    )
    checks["no_backtrack_ablation_fails_reversal"] = all(
        cell(task, "mechanism_reversal", "radp_v1_no_backtrack")["backtrack_success_rate"] == 0.0
        for task in TASKS
    )
    checks["strict_reliability_improvement_exists"] = (
        checks["radp_rejects_spurious_equal_total"]
        and checks["radp_backtracks_reversal_equal_total"]
    )
    checks["verification_boundary_explicit"] = True
    checks["independent_physical_channel_available"] = False
    checks["all_checks_pass"] = all(
        value for key, value in checks.items()
        if key != "independent_physical_channel_available"
    )

    result = {
        "schema_version": "matched_reliability_cost_audit.v1",
        "model": "qwen3-235b-a22b-instruct-2507",
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "primary_per_run": EXPECTED_PRIMARY,
        "total_budget_cap": TOTAL_BUDGET,
        "replay_conditions": list(CONDITIONS),
        "replay_policies": list(REPLAY_POLICIES),
        "inputs": inputs,
        "cells": cells,
        "checks": checks,
        "pareto_front": bool(checks["all_checks_pass"]),
        "independent_verification_channel": False,
        "claim_boundary": (
            "Offline evaluator-side corruption replay on completed matched trajectories. "
            "The hidden source condition is not exposed to policies. This establishes a "
            "reliability-cost diagnostic, not an independent physical verification result, "
            "official leaderboard result, or external benchmark result."
        ),
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "checks": checks}, indent=2, ensure_ascii=False))
    return 0 if result["pareto_front"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
