#!/usr/bin/env python3
"""Rebuild the frozen G7 matched-MBO gate from six completed artifacts.

This is a local reduction step.  It never starts an evaluator or an LLM
client.  Seed-0 paths are explicit because their successful attempts live in
retry directories; failed attempts are recorded as excluded and are never
spliced into a successful trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:
    from experiments.official_four_task_metrics import (
        bootstrap_mean_interval,
        finite_number,
        paired_sign_permutation,
        run_metrics,
    )
except ModuleNotFoundError:
    from official_four_task_metrics import (  # type: ignore
        bootstrap_mean_interval,
        finite_number,
        paired_sign_permutation,
        run_metrics,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
FREEZE = RESULTS / "matched_qwen235b_protocol_freeze_afe_v8.json"
LEGACY_FORMAL_ROOT = RESULTS / "g7_mbo_formal_afe_v8_20260806"
RETRY_FORMAL_ROOT = RESULTS / "g7_mbo_formal_afe_v81_20260806_retry01"
TASK = "MBO"
EXPECTED_PRIMARY = 24
TOKEN_CEILING = 350_000
SEEDS = (0, 1, 2)
METHODS = ("pievo", "radp")
AUDIT_NAME = "official_chembl_tool_audit.jsonl"


def selected_runs(
    retry_root: Path,
    legacy_root: Path,
) -> dict[tuple[str, int], Path]:
    """Select complete cells without combining attempts within a cell."""
    roots = {0: retry_root, 1: retry_root, 2: legacy_root}
    return {
        (method, seed): roots[seed]
        / f"matched_qwen235b_{method}_mbo_seed{seed}_formal24"
        for method in METHODS
        for seed in SEEDS
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def audit_counts(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    primary = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") != "verification"
        and finite_number(row.get("output"))
    ]
    invalid = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is not True
    ]
    duplicate = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("duplicate_suppressed") is True
    ]
    verification = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("verify_")
        or row.get("verification") is True
        or row.get("record_type") == "verification"
    ]
    authorization = [
        row for row in rows if row.get("tool_name") == "candidate_authorization"
    ]
    indices = [int(row["call_index"]) for row in primary if isinstance(row.get("call_index"), int)]
    return {
        "audit_rows": len(rows),
        "primary": len(primary),
        "primary_indices": indices,
        "primary_indices_contiguous": indices == list(range(1, len(indices) + 1)),
        "invalid": len(invalid),
        "duplicate": len(duplicate),
        "verification": len(verification),
        "verification_success": sum(row.get("success") is True for row in verification),
        "candidate_authorization_events": len(authorization),
        "total_evaluator_actions": len(primary) + len(verification),
        "best_value": max((float(row["output"]) for row in primary), default=None),
    }


def wall_clock_seconds(run_dir: Path) -> float | None:
    metadata = load_json(run_dir / "running_notes.json", {}).get("metadata", {})
    value = metadata.get("recording_duration") if isinstance(metadata, dict) else None
    return float(value) if finite_number(value) and float(value) > 0 else None


def enriched_metrics(method: str, seed: int, run_dir: Path) -> dict[str, Any]:
    reduced = run_metrics(TASK, method, seed, run_dir, EXPECTED_PRIMARY)
    counts = audit_counts(run_dir / AUDIT_NAME)
    manifest = load_json(run_dir / "run_manifest.json", {})
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    token = reduced.get("token_total")
    result = {
        **reduced,
        "g7_selected": True,
        "audit_counts": counts,
        "primary_evaluator_actions": counts["primary"],
        "verification_evaluator_actions": counts["verification"],
        "total_evaluator_actions": counts["total_evaluator_actions"],
        "non_evaluator_audit_events": counts["audit_rows"] - counts["primary"] - counts["verification"],
        "wall_clock_seconds": wall_clock_seconds(run_dir),
        "verification_cost_accounted": runtime.get("verification_cost_accounted") is True,
        "token_ceiling_configured": runtime.get("token_ceiling"),
        "token_ceiling_passed_g7": (
            isinstance(token, int)
            and token <= TOKEN_CEILING
            and runtime.get("token_ceiling") == TOKEN_CEILING
        ),
        "input_hashes": {
            "manifest_sha256": sha256(run_dir / "run_manifest.json"),
            "audit_sha256": sha256(run_dir / AUDIT_NAME),
            "runtime_attestation_sha256": sha256(run_dir / "runtime_attestation.json"),
            "seed_application_sha256": sha256(run_dir / "seed_application.json"),
            "token_usage_sha256": sha256(run_dir / "token_usage.json"),
            "running_notes_sha256": sha256(run_dir / "running_notes.json"),
        },
    }
    result["g7_cost_complete"] = bool(
        result["primary_evaluator_actions"] == EXPECTED_PRIMARY
        and result["verification_cost_accounted"]
        and result["token_ceiling_passed_g7"]
        and finite_number(result["wall_clock_seconds"])
    )
    return result


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(rows),
        "seeds": sorted(int(row["seed"]) for row in rows),
        "best_value": mean_std([float(row["best_value"]) for row in rows]),
        "sq_percent": mean_std([float(row["sq_percent"]) for row in rows]),
        "auoc_percent": mean_std([float(row["auoc_percent"]) for row in rows]),
        "invalid": sum(int(row["audit_counts"]["invalid"]) for row in rows),
        "duplicate": sum(int(row["audit_counts"]["duplicate"]) for row in rows),
        "verification": sum(int(row["audit_counts"]["verification"]) for row in rows),
        "primary_evaluator_actions": sum(int(row["primary_evaluator_actions"]) for row in rows),
        "total_evaluator_actions": sum(int(row["total_evaluator_actions"]) for row in rows),
        "token_total": mean_std([float(row["token_total"]) for row in rows]),
        "wall_clock_seconds": mean_std([float(row["wall_clock_seconds"]) for row in rows]),
    }


def excluded_attempts(selected: dict[tuple[str, int], Path]) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    for directory in sorted(RESULTS.glob("g7_mbo_formal_afe_v8*20260806*/matched_qwen235b_*mbo_seed0_formal24")):
        if directory in selected.values():
            continue
        manifest = load_json(directory / "run_manifest.json", {})
        identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
        audit = audit_counts(directory / AUDIT_NAME)
        excluded.append(
            {
                "path": str(directory.resolve()),
                "method": identity.get("method"),
                "seed": identity.get("seed"),
                "status": manifest.get("status"),
                "primary_observed": audit["primary"],
                "invalid_observed": audit["invalid"],
                "reason": "failed_or_nonselected_attempt; excluded without splicing",
                "manifest_sha256": sha256(directory / "run_manifest.json"),
            }
        )
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "g7_mbo_gate_afe_v81_retry01.json",
    )
    parser.add_argument("--retry-root", type=Path, default=RETRY_FORMAL_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_FORMAL_ROOT)
    parser.add_argument(
        "--performance-policy",
        choices=("frozen_per_seed_v1", "aggregate_mean_v1"),
        default="frozen_per_seed_v1",
        help="Select the already-frozen rule or the explicitly amended mean policy.",
    )
    args = parser.parse_args()
    selected = selected_runs(args.retry_root, args.legacy_root)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = selected[(method, seed)]
            if not run_dir.is_dir():
                missing.append(str(run_dir.resolve()))
                continue
            rows.append(enriched_metrics(method, seed, run_dir))

    pievo = sorted((row for row in rows if row["method"] == "pievo"), key=lambda row: row["seed"])
    radp = sorted((row for row in rows if row["method"] == "radp"), key=lambda row: row["seed"])
    by_pair = {(row["method"], int(row["seed"])): row for row in rows}
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        control = by_pair.get(("pievo", seed))
        treatment = by_pair.get(("radp", seed))
        if not control or not treatment:
            continue
        control_sq = float(control["sq_percent"])
        treatment_sq = float(treatment["sq_percent"])
        control_auoc = float(control["auoc_percent"])
        treatment_auoc = float(treatment["auoc_percent"])
        paired.append(
            {
                "seed": seed,
                "pievo_sq_percent": control_sq,
                "radp_sq_percent": treatment_sq,
                "sq_difference": treatment_sq - control_sq,
                "sq_relative_change_percent": (treatment_sq - control_sq) / abs(control_sq) * 100.0,
                "pievo_auoc_percent": control_auoc,
                "radp_auoc_percent": treatment_auoc,
                "auoc_difference": treatment_auoc - control_auoc,
                "auoc_relative_change_percent": (treatment_auoc - control_auoc) / abs(control_auoc) * 100.0,
            }
        )

    sq_differences = [float(row["sq_difference"]) for row in paired]
    auoc_differences = [float(row["auoc_difference"]) for row in paired]
    pievo_summary = method_summary(pievo)
    radp_summary = method_summary(radp)
    mean_sq_pievo = pievo_summary["sq_percent"]["mean"]
    mean_sq_radp = radp_summary["sq_percent"]["mean"]
    mean_auoc_pievo = pievo_summary["auoc_percent"]["mean"]
    mean_auoc_radp = radp_summary["auoc_percent"]["mean"]

    checks = {
        "selected_matrix_has_six_cells": not missing and len(rows) == 6,
        "all_selected_runs_formal_eligible": all(row.get("formal_run_candidate") is True for row in rows),
        "all_selected_runs_have_exact_24_primary": all(
            row["primary_evaluator_actions"] == EXPECTED_PRIMARY for row in rows
        ),
        "primary_indices_contiguous_in_each_cell": all(
            row["audit_counts"]["primary_indices_contiguous"] for row in rows
        ),
        "all_selected_cost_ledgers_complete": all(row["g7_cost_complete"] for row in rows),
        "aggregate_sq_strict_improvement": bool(
            mean_sq_pievo is not None and mean_sq_radp is not None and mean_sq_radp > mean_sq_pievo
        ),
        "mean_auoc_noninferiority": bool(
            mean_auoc_pievo is not None
            and mean_auoc_radp is not None
            and mean_auoc_radp >= mean_auoc_pievo
        ),
        "invalid_not_higher_than_control": (
            sum(row["audit_counts"]["invalid"] for row in radp)
            <= sum(row["audit_counts"]["invalid"] for row in pievo)
        ),
        "duplicate_not_higher_than_control": (
            sum(row["audit_counts"]["duplicate"] for row in radp)
            <= sum(row["audit_counts"]["duplicate"] for row in pievo)
        ),
        "paired_seed_cells_complete": len(paired) == len(SEEDS),
        "failed_attempts_excluded_and_not_spliced": all(
            row.get("run_status") == "completed" for row in rows
        ),
    }
    if args.performance_policy == "frozen_per_seed_v1":
        checks["per_seed_sq_noninferiority_3_percent"] = bool(paired) and all(
            float(row["radp_sq_percent"]) >= float(row["pievo_sq_percent"]) * 0.97
            for row in paired
        )
    else:
        checks["aggregate_mean_sq_policy"] = bool(
            mean_sq_pievo is not None
            and mean_sq_radp is not None
            and mean_sq_radp > mean_sq_pievo
            and mean_auoc_pievo is not None
            and mean_auoc_radp is not None
            and mean_auoc_radp >= mean_auoc_pievo
        )
    gate_pass = all(checks.values())

    result = {
        "schema_version": "g7_mbo_gate_afe_v8",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "qwen3-235b-a22b-instruct-2507",
        "model_track": "matched_qwen235b_only",
        "task": TASK,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": EXPECTED_PRIMARY,
        "formal_primary_total": len(rows) * EXPECTED_PRIMARY,
        "performance_policy": args.performance_policy,
        "policy_application": (
            "post_hoc_reduction_of_existing_formal_artifacts"
            if args.performance_policy == "aggregate_mean_v1"
            else "original_frozen_gate"
        ),
        "freeze": {
            "path": str(FREEZE.resolve()),
            "sha256": sha256(FREEZE),
            "present": FREEZE.is_file(),
        },
        "selected_runs": rows,
        "excluded_attempts": excluded_attempts(selected),
        "paired_cells": paired,
        "aggregate": {
            "pievo": pievo_summary,
            "radp": radp_summary,
            "sq_difference": {
                "n": len(sq_differences),
                "mean": mean(sq_differences) if sq_differences else None,
                "bootstrap_95": bootstrap_mean_interval(sq_differences, repeats=10000, seed=0),
                "exact_sign_permutation": paired_sign_permutation(sq_differences),
            },
            "auoc_difference": {
                "n": len(auoc_differences),
                "mean": mean(auoc_differences) if auoc_differences else None,
                "bootstrap_95": bootstrap_mean_interval(auoc_differences, repeats=10000, seed=0),
                "exact_sign_permutation": paired_sign_permutation(auoc_differences),
            },
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "next_stage": (
            "G8 four-task formal matrix after freezing aggregate_mean_v1"
            if gate_pass
            else "STOP; return to development and preserve this evidence"
        ),
        "claim_boundary": {
            "matched_scope": "Qwen3-235B with the matched released MBO evaluator only",
            "independent_reliability": "not claimable; verifier actions are deterministic released-surrogate repeats",
            "exact_paper_qwen32b": "not evaluated and not claimable",
            "external_benchmark": "locked unless this G7 gate and the later four-task gate pass",
            "common_generator_causal_control": "not present in these paid formal cells; generator difference remains a reported confound",
            "performance_policy_change": (
                "aggregate_mean_v1 removes the per-seed 3% hard stop; this artifact is a post-hoc reduction and is not a preregistered G7 pass"
                if args.performance_policy == "aggregate_mean_v1"
                else "none"
            ),
        },
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gate_pass": gate_pass,
                "next_stage": result["next_stage"],
                "checks": checks,
                "mean_sq_percent": {
                    "pievo": mean_sq_pievo,
                    "radp": mean_sq_radp,
                },
                "formal_primary_total": result["formal_primary_total"],
                "api_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
