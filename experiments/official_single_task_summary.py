#!/usr/bin/env python3
"""Reduce one completed matched task without making an external claim."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:
    from experiments.official_four_task_metrics import (
        bootstrap_mean_interval,
        paired_sign_permutation,
        run_metrics,
    )
except ModuleNotFoundError:
    from official_four_task_metrics import (  # type: ignore
        bootstrap_mean_interval,
        paired_sign_permutation,
        run_metrics,
    )


METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def wall_clock_seconds(run_dir: Path) -> float | None:
    metadata = load_json(run_dir / "running_notes.json").get("metadata", {})
    value = metadata.get("recording_duration") if isinstance(metadata, dict) else None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("NHO", "MBO", "SPO", "TMC"))
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-primary", type=int, default=24)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = args.results_root / (
                f"matched_qwen235b_{method}_{args.task.lower()}_seed{seed}_formal24"
            )
            if not run_dir.is_dir():
                missing.append(str(run_dir.resolve()))
                continue
            reduced = run_metrics(args.task, method, seed, run_dir, args.expected_primary)
            reduced["wall_clock_seconds"] = wall_clock_seconds(run_dir)
            reduced["manifest_sha256"] = sha256(run_dir / "run_manifest.json")
            reduced["audit_sha256"] = sha256(
                run_dir / {
                    "NHO": "official_nho_tool_audit.jsonl",
                    "MBO": "official_chembl_tool_audit.jsonl",
                    "SPO": "official_spo_tool_audit.jsonl",
                    "TMC": "official_tmc_tool_audit.jsonl",
                }[args.task]
            )
            rows.append(reduced)

    by_key = {(row["method"], int(row["seed"])): row for row in rows}
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        control = by_key.get(("pievo", seed))
        treatment = by_key.get(("radp", seed))
        if not control or not treatment:
            continue
        paired.append(
            {
                "seed": seed,
                "sq_difference": treatment["sq_percent"] - control["sq_percent"],
                "auoc_difference": treatment["auoc_percent"] - control["auoc_percent"],
                "pievo_sq_percent": control["sq_percent"],
                "radp_sq_percent": treatment["sq_percent"],
                "pievo_auoc_percent": control["auoc_percent"],
                "radp_auoc_percent": treatment["auoc_percent"],
            }
        )

    summaries: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {
            "runs": len(method_rows),
            "seeds": sorted(int(row["seed"]) for row in method_rows),
            "best_value": mean_std([row["best_value"] for row in method_rows]),
            "sq_percent": mean_std([row["sq_percent"] for row in method_rows]),
            "auoc_percent": mean_std([row["auoc_percent"] for row in method_rows]),
            "invalid_attempts": sum(row["invalid_attempts"] for row in method_rows),
            "invalid_rate": mean_std([row["invalid_rate"] for row in method_rows]),
            "duplicate_suppressed": sum(row["duplicate_suppressed"] for row in method_rows),
            "duplicate_rate": mean_std([row["duplicate_rate"] for row in method_rows]),
            "verification_calls": sum(row["verification_calls"] for row in method_rows),
            "total_evaluator_calls": sum(row["total_evaluator_calls"] for row in method_rows),
            "token_total": mean_std([row["token_total"] for row in method_rows]),
            "wall_clock_seconds": mean_std(
                [row["wall_clock_seconds"] for row in method_rows if row["wall_clock_seconds"] is not None]
            ),
        }

    sq_differences = [row["sq_difference"] for row in paired]
    auoc_differences = [row["auoc_difference"] for row in paired]
    checks = {
        "six_paired_cells_complete": not missing and len(paired) == 3,
        "all_runs_formal_eligible": bool(rows) and all(row["formal_run_candidate"] for row in rows),
        "all_primary_budgets_complete": bool(rows) and all(
            row["primary_evaluations"] == args.expected_primary for row in rows
        ),
        "mean_sq_strictly_higher": bool(
            summaries["radp"]["sq_percent"]["mean"] is not None
            and summaries["pievo"]["sq_percent"]["mean"] is not None
            and summaries["radp"]["sq_percent"]["mean"] > summaries["pievo"]["sq_percent"]["mean"]
        ),
        "mean_auoc_noninferior": bool(
            summaries["radp"]["auoc_percent"]["mean"] is not None
            and summaries["pievo"]["auoc_percent"]["mean"] is not None
            and summaries["radp"]["auoc_percent"]["mean"] >= summaries["pievo"]["auoc_percent"]["mean"]
        ),
        "mean_invalid_rate_not_higher": bool(
            summaries["radp"]["invalid_rate"]["mean"] is not None
            and summaries["pievo"]["invalid_rate"]["mean"] is not None
            and summaries["radp"]["invalid_rate"]["mean"] <= summaries["pievo"]["invalid_rate"]["mean"]
        ),
    }
    result = {
        "schema_version": "official_single_task_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "model": "qwen3-235b-a22b-instruct-2507",
        "results_root": str(args.results_root.resolve()),
        "expected_primary_per_cell": args.expected_primary,
        "methods": summaries,
        "paired_cells": paired,
        "paired_sq_difference": {
            "mean": mean(sq_differences) if sq_differences else None,
            "bootstrap_95": bootstrap_mean_interval(sq_differences),
            "exact_sign_permutation": paired_sign_permutation(sq_differences),
        },
        "paired_auoc_difference": {
            "mean": mean(auoc_differences) if auoc_differences else None,
            "bootstrap_95": bootstrap_mean_interval(auoc_differences),
            "exact_sign_permutation": paired_sign_permutation(auoc_differences),
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
        "missing_runs": missing,
        "claim_boundary": {
            "matched_released_evaluator_only": True,
            "complete_four_task_sota": False,
            "independent_mbo_held_out_verifier": False,
            "equal_total_evaluator_cost": False,
            "apd_claimable": False,
            "interpretation": "Three-seed matched SPO interim comparison; not a complete SOTA claim.",
        },
        "run_metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate_pass": result["gate_pass"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
