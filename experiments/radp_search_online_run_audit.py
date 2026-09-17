#!/usr/bin/env python3
"""Audit one online RADP-Search/PiEvo run without inventing APD."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


MU_REF = {"MBO": 6.5}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trajectory_metrics(rows: list[dict[str, Any]], mu_ref: float) -> dict[str, Any]:
    values = [float(row["value"]) for row in rows]
    finite = all(math.isfinite(value) for value in values)
    running_max: list[float] = []
    best = -math.inf
    for value in values:
        best = max(best, value)
        running_max.append(best)
    sq = (max(values) / mu_ref * 100.0) if values else None
    auoc = (sum(running_max) / (len(values) * mu_ref) * 100.0) if values else None
    return {
        "primary_evaluations": len(values),
        "all_values_finite": finite,
        "best_value": max(values) if values else None,
        "sq_percent": sq,
        "auoc_percent": auoc,
    }


def relative_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return abs(value - reference) / abs(reference) * 100.0


def relative_drop(value: float | None, reference: float | None) -> float | None:
    """Return only degradation; improvements are not counted as a drop."""
    if value is None or reference in (None, 0):
        return None
    return max(0.0, (reference - value) / abs(reference) * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument(
        "--control-run",
        type=Path,
        default=None,
        help=(
            "Matched-protocol control run used for non-inferiority and cost "
            "comparisons. Defaults to --reference-run for legacy audits."
        ),
    )
    parser.add_argument("--task", default="MBO")
    parser.add_argument("--expected-primary", type=int, default=24)
    parser.add_argument("--status", choices=["formal", "partial", "aborted", "failed"], default="partial")
    parser.add_argument("--include-in-statistics", action="store_true")
    parser.add_argument("--termination", default="unspecified")
    args = parser.parse_args()

    run_dir = args.run_dir
    reference_dir = args.reference_run
    control_dir = args.control_run or reference_dir
    mu_ref = MU_REF[args.task]
    experiment_rows = load_json(run_dir / "experiment.json", [])
    if not isinstance(experiment_rows, list):
        experiment_rows = []
    experiment_rows = [row for row in experiment_rows if isinstance(row, dict)]
    tool_rows = load_jsonl(run_dir / "official_chembl_tool_audit.jsonl")
    gate_audit_path = run_dir / "source_gate_audit.jsonl"
    if not gate_audit_path.exists():
        gate_audit_path = run_dir / "radp_source_gate_audit.jsonl"
    gate_rows = load_jsonl(gate_audit_path)

    primary_tool_rows = [
        row
        for row in tool_rows
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("success") is True
        and not row.get("duplicate_suppressed", False)
        and row.get("call_index") is not None
    ]
    failed_tool_rows = [
        row
        for row in tool_rows
        if row.get("tool_name") == "evaluate_chembl" and row.get("success") is False
    ]
    duplicate_rows = [
        row
        for row in tool_rows
        if row.get("tool_name") == "evaluate_chembl" and row.get("duplicate_suppressed") is True
    ]
    verification_rows = [row for row in tool_rows if row.get("tool_name") == "verify_chembl"]

    # The evaluator audit is the source of truth for the primary trajectory.
    # The submission-derived experiment.json can lag by one row when the
    # evaluator cap is reached during a tool-call turn.
    values = [row for row in experiment_rows if isinstance(row.get("value"), (int, float))]
    primary_values = [
        {"value": float(row["output"])}
        for row in primary_tool_rows
        if isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
    ]
    metrics = trajectory_metrics(primary_values, mu_ref)
    primary_indices = sorted(int(row["call_index"]) for row in primary_tool_rows)
    primary_sequence_contiguous = primary_indices == list(range(1, len(primary_indices) + 1))
    gate_decisions = Counter(row.get("source_decision") for row in gate_rows)
    gate_behaviors = Counter(row.get("gate_behavior") for row in gate_rows)

    corrupted_primary = [
        row for row in primary_tool_rows if row.get("corruption_applied") is True
    ]
    corrupted_inputs = {str(row.get("input")) for row in corrupted_primary}
    corrupted_gate_rows = [
        row for row in gate_rows if str(row.get("hypothesis")) in corrupted_inputs
    ]
    corrupted_gate_rejections = [
        row
        for row in corrupted_gate_rows
        if row.get("update_allowed") is False
        or row.get("source_decision") == "surrogate_channels_disagree"
    ]
    corrupted_input_set = set(corrupted_inputs)
    rejected_corrupted_inputs = {
        str(row.get("hypothesis"))
        for row in corrupted_gate_rejections
        if row.get("verified_observation") is not None
    }
    corruption_replacements = {
        str(row.get("hypothesis")): float(row["verified_observation"])
        for row in corrupted_gate_rejections
        if row.get("verified_observation") is not None
    }
    corrected_primary_values = [
        {
            "value": corruption_replacements.get(
                str(row.get("input")), float(row["output"])
            )
        }
        for row in primary_tool_rows
        if isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
    ]
    corrected_metrics = trajectory_metrics(corrected_primary_values, mu_ref)

    reference_tool_rows = load_jsonl(reference_dir / "official_chembl_tool_audit.jsonl")
    reference_primary = [
        row
        for row in reference_tool_rows
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("success") is True
        and not row.get("duplicate_suppressed", False)
        and row.get("call_index") is not None
    ]
    reference_primary_values = [
        {"value": float(row["output"])}
        for row in reference_primary
        if isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
    ]
    reference_rows = load_json(reference_dir / "experiment.json", [])
    if not isinstance(reference_rows, list):
        reference_rows = []
    reference_rows = [row for row in reference_rows if isinstance(row, dict) and isinstance(row.get("value"), (int, float))]
    reference_metrics = trajectory_metrics(
        reference_primary_values or reference_rows,
        mu_ref,
    )

    control_tool_rows = load_jsonl(control_dir / "official_chembl_tool_audit.jsonl")
    control_primary = [
        row
        for row in control_tool_rows
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("success") is True
        and not row.get("duplicate_suppressed", False)
        and row.get("call_index") is not None
    ]
    control_primary_values = [
        {"value": float(row["output"])}
        for row in control_primary
        if isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
    ]
    control_rows = load_json(control_dir / "experiment.json", [])
    if not isinstance(control_rows, list):
        control_rows = []
    control_rows = [
        row
        for row in control_rows
        if isinstance(row, dict) and isinstance(row.get("value"), (int, float))
    ]
    control_metrics = trajectory_metrics(
        control_primary_values or control_rows,
        mu_ref,
    )

    primary_complete = (
        len(primary_tool_rows) == args.expected_primary
        and len(primary_values) == args.expected_primary
        and primary_sequence_contiguous
    )
    sq_delta = relative_delta(
        corrected_metrics["sq_percent"], control_metrics["sq_percent"]
    )
    auoc_delta = relative_delta(
        corrected_metrics["auoc_percent"], control_metrics["auoc_percent"]
    )
    sq_drop = relative_drop(
        corrected_metrics["sq_percent"], control_metrics["sq_percent"]
    )
    auoc_drop = relative_drop(
        corrected_metrics["auoc_percent"], control_metrics["auoc_percent"]
    )
    three_percent_pass = (
        sq_drop is not None
        and auoc_drop is not None
        and sq_drop <= 3.0
        and auoc_drop <= 3.0
    )
    invalid_rate_denominator = len(primary_tool_rows) + len(failed_tool_rows)
    invalid_rate = (
        len(failed_tool_rows) / invalid_rate_denominator if invalid_rate_denominator else None
    )
    reference_failed = [
        row
        for row in reference_tool_rows
        if row.get("tool_name") == "evaluate_chembl" and row.get("success") is False
    ]
    reference_denominator = len(reference_primary) + len(reference_failed)
    reference_invalid_rate = (
        len(reference_failed) / reference_denominator if reference_denominator else None
    )
    control_failed = [
        row
        for row in control_tool_rows
        if row.get("tool_name") == "evaluate_chembl" and row.get("success") is False
    ]
    control_denominator = len(control_primary) + len(control_failed)
    control_invalid_rate = (
        len(control_failed) / control_denominator if control_denominator else None
    )
    invalid_rate_not_higher_than_control = (
        invalid_rate is not None
        and control_invalid_rate is not None
        and invalid_rate <= control_invalid_rate
    )
    controlled_corruption_gate_pass = (
        not corrupted_primary
        or rejected_corrupted_inputs >= corrupted_input_set
    )

    checks = {
        "experiment_artifact_present": (run_dir / "experiment.json").exists(),
        "official_tool_audit_present": (run_dir / "official_chembl_tool_audit.jsonl").exists(),
        "primary_budget_complete": primary_complete,
        "experiment_rows_match_primary_budget": len(values) == len(primary_values),
        "primary_outputs_finite": metrics["all_values_finite"],
        "primary_indices_contiguous": primary_sequence_contiguous,
        "verification_calls_present": len(verification_rows) > 0,
        "gate_audit_present": gate_audit_path.exists(),
        "apd_exact_available": False,
        "three_percent_sq_auoc_control": three_percent_pass,
        "invalid_rate_not_higher_than_control": invalid_rate_not_higher_than_control,
        "controlled_corruption_gate_pass": controlled_corruption_gate_pass,
    }
    formal_result = args.status == "formal" and primary_complete and all(
        checks[key]
        for key in (
            "experiment_artifact_present",
            "official_tool_audit_present",
            "primary_outputs_finite",
            "primary_indices_contiguous",
            "three_percent_sq_auoc_control",
            "invalid_rate_not_higher_than_control",
            "controlled_corruption_gate_pass",
        )
    )
    include_in_statistics = bool(args.include_in_statistics and formal_result)
    checks["formal_session_clean"] = formal_result
    checks["include_in_statistics"] = include_in_statistics

    audit = {
        "schema_version": "radp_search_online_run_audit.v1",
        "scope": {
            "run_dir": str(run_dir),
            "reference_run_dir": str(reference_dir),
            "control_run_dir": str(control_dir),
            "task": args.task,
            "mu_ref": mu_ref,
            "expected_primary": args.expected_primary,
            "termination": args.termination,
        },
        "counts": {
            "experiment_rows": len(experiment_rows),
            "primary_successful_evaluations": len(primary_tool_rows),
            "invalid_or_failed_attempts": len(failed_tool_rows),
            "duplicate_suppressed_calls": len(duplicate_rows),
            "verification_calls": len(verification_rows),
            "gate_audit_rows": len(gate_rows),
        },
        "metrics": metrics,
        "corrected_metrics": corrected_metrics,
        "reference_metrics": reference_metrics,
        "control_metrics": control_metrics,
        "comparison": {
            "sq_relative_delta_percent": sq_delta,
            "auoc_relative_delta_percent": auoc_delta,
            "sq_relative_drop_percent": sq_drop,
            "auoc_relative_drop_percent": auoc_drop,
            "three_percent_sq_auoc_pass": three_percent_pass,
            "metric_source": "verified_corrected_trajectory",
            "invalid_rate": invalid_rate,
            "control_invalid_rate": control_invalid_rate,
            "reference_invalid_rate": reference_invalid_rate,
            "invalid_rate_not_higher_than_control": invalid_rate_not_higher_than_control,
            "invalid_rate_not_higher_than_reference": (
                invalid_rate is not None
                and reference_invalid_rate is not None
                and invalid_rate <= reference_invalid_rate
            ),
        },
        "gate": {
            "source_decisions": dict(gate_decisions),
            "behaviors": dict(gate_behaviors),
            "corrupted_primary_observations": len(corrupted_primary),
            "corrupted_gate_rows": len(corrupted_gate_rows),
            "corrupted_gate_rejections": len(corrupted_gate_rejections),
            "rejected_corrupted_inputs": len(rejected_corrupted_inputs),
            "controlled_corruption_gate_pass": controlled_corruption_gate_pass,
        },
        "checks": checks,
        "status": args.status,
        "formal_result": formal_result,
        "include_in_statistics": include_in_statistics,
        "apd_status": "not_available_exact_definition_and_evaluator_missing",
        "interpretation": (
            "SQ/AUOC are computed from the retained online trajectory. APD is intentionally not claimed. "
            "The formal comparison uses --control-run when provided; --reference-run remains a legacy/context "
            "run. A 3% comparison is a development control only and is not a paired causal estimate when the "
            "online candidate sequence differs from the matched control sequence. Controlled corruption gate "
            "coverage is reported separately and is required before a robustness claim."
        ),
    }
    status = {
        "schema_version": "radp_search_online_run_status.v1",
        "status": args.status,
        "formal_result": formal_result,
        "include_in_statistics": include_in_statistics,
        "primary_budget_complete": primary_complete,
        "termination": args.termination,
        "reason": (
            f"Run marked {args.status}; termination={args.termination}."
            if args.status in {"partial", "aborted"}
            else "Status generated by the online-run audit."
        ),
        "apd_status": "not_available_exact_definition_and_evaluator_missing",
    }
    (run_dir / "online_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (run_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "metrics": metrics, "comparison": audit["comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
