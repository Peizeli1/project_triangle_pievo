#!/usr/bin/env python3
"""Compute the frozen four-task metrics and official unlock decision.

The evaluator audit is authoritative for primary trajectories.  This script
does not call an LLM, does not repair missing runs, and never treats a
surrogate smoke test as a formal result.  It can therefore be run before and
after API collection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

try:
    from experiments.formal_runtime_contract import (
        attestation_is_formal,
        formal_approval_is_valid,
    )
    from experiments.formal_usage import last_cumulative_usage, read_cumulative_usage
except ImportError:
    from formal_runtime_contract import attestation_is_formal, formal_approval_is_valid
    from formal_usage import last_cumulative_usage, read_cumulative_usage


TASKS = ("NHO", "MBO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)
RADP_ALGORITHM_IDS = frozenset(
    {
        "radp_search_active_v1",
        "radp_search_afe_v1",
        "radp_search_afe_v8",
        "radp_search_afe_spo_v1",
        "radp_search_afe_v9",
        "radp_search_afe_spo_v2",
        "radp_search_afe_spo_v3",
    }
)
MU_REF = {"NHO": 2.0, "MBO": 6.5, "SPO": 298.5, "TMC": 493.8}
BOOTSTRAP_REPEATS = 10000
AUDIT_NAMES = {
    "NHO": "official_nho_tool_audit.jsonl",
    "MBO": "official_chembl_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
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
    except (OSError, ValueError):
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def primary_rows(task: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select successful non-duplicate evaluator observations."""
    selected: list[dict[str, Any]] = []
    for row in rows:
        tool_name = str(row.get("tool_name", ""))
        if not tool_name.startswith("evaluate_"):
            continue
        if row.get("success") is not True or row.get("duplicate_suppressed") is True:
            continue
        if row.get("record_type") in {"duplicate_suppressed", "verification"}:
            continue
        if not finite_number(row.get("output")):
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: int(row.get("call_index", 0)))


def invalid_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is not True
    ]


def duplicate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("duplicate_suppressed") is True
    ]


def verification_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("tool_name", "")).startswith("verify_") or row.get("verification") is True]


def token_total(value: Any) -> int:
    return int(last_cumulative_usage(value)["total"] or 0)


def seed_proof_valid(value: Any, *, seed: int, algorithm_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    client_seeds = value.get("llm_client_seeds")
    return bool(
        value.get("schema_version") == "pievo_seed_application.v1"
        and value.get("seed") == seed
        and value.get("algorithm_id") == algorithm_id
        and value.get("seed_applied") is True
        and value.get("llm_clients_seeded") is True
        and isinstance(client_seeds, list)
        and client_seeds
        and all(client_seed == seed for client_seed in client_seeds)
    )


def radp_runtime_audit_valid(path: Path, *, task: str) -> bool:
    expected_provider = f"official_{task.lower()}"
    for row in load_jsonl(path):
        if (
            row.get("schema_version") == "radp_runtime_audit.v1"
            and row.get("event") == "runtime_start"
            and row.get("algorithm_id") in RADP_ALGORITHM_IDS
            and row.get("mode") == "active"
            and row.get("provider") == expected_provider
            and row.get("source_gate_enabled") is True
            and row.get("revision_protocol_enabled") is True
            and row.get("backtrack_enabled") is True
        ):
            return True
    return False


def trajectory_metrics(task: str, rows: list[dict[str, Any]], expected_primary: int) -> dict[str, Any]:
    values = [float(row["output"]) for row in rows]
    mu_ref = MU_REF[task]
    running_best: list[float] = []
    current = -math.inf
    for value in values:
        current = max(current, value)
        running_best.append(current)
    invalid = 0
    duplicate = 0
    verification = 0
    return {
        "primary_evaluations": len(values),
        "expected_primary": expected_primary,
        "primary_budget_complete": len(values) == expected_primary,
        "all_values_finite": all(math.isfinite(value) for value in values),
        "best_value": max(values) if values else None,
        "sq_percent": (max(values) / mu_ref * 100.0) if values else None,
        "auoc_percent": (mean(running_best) / mu_ref * 100.0) if values else None,
        "mu_ref": mu_ref,
        "invalid_attempts": invalid,
        "duplicate_suppressed": duplicate,
        "verification_calls": verification,
    }


def run_metrics(task: str, method: str, seed: int, run_dir: Path, expected_primary: int) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json(manifest_path, {})
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    claim_boundary = manifest.get("claim_boundary", {}) if isinstance(manifest, dict) else {}
    manifest_schema_ok = manifest.get("schema_version") == "matched_qwen235b_run_manifest.v3"
    manifest_identity_ok = bool(
        isinstance(identity, dict)
        and identity.get("task") == task
        and identity.get("method") == method
        and identity.get("seed") == seed
        and identity.get("primary_budget") == expected_primary
        and identity.get("model") == "qwen3-235b-a22b-instruct-2507"
        and identity.get("track") == "matched_qwen235b"
        and identity.get("stage") == "formal"
    )
    official_task_assets_verified = bool(
        isinstance(claim_boundary, dict)
        and claim_boundary.get("official_task_assets_verified") is True
    )
    matched_evaluator_assets_verified = bool(
        isinstance(claim_boundary, dict)
        and claim_boundary.get("matched_evaluator_assets_verified") is True
    )
    audit_path = run_dir / AUDIT_NAMES[task]
    rows = load_jsonl(audit_path)
    primary = primary_rows(task, rows)
    invalid = invalid_rows(rows)
    duplicates = duplicate_rows(rows)
    verification = verification_rows(rows)
    final_evidence = manifest.get("final_evidence", {}) if isinstance(manifest, dict) else {}
    manifest_audit = final_evidence.get("audit", {}) if isinstance(final_evidence, dict) else {}
    if not isinstance(manifest_audit, dict):
        manifest_audit = {}

    def audited_count(name: str, fallback: int) -> int:
        value = manifest_audit.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return fallback

    effective_invalid = audited_count("effective_invalid", len(invalid))
    effective_duplicate = audited_count("effective_duplicate", len(duplicates))
    prefilter_invalid = audited_count("prefilter_invalid", 0)
    prefilter_duplicate = audited_count("prefilter_duplicate", 0)
    afe_controller_invalid = audited_count("afe_controller_invalid", 0)
    afe_controller_duplicate = audited_count("afe_controller_duplicate", 0)
    runtime_path = run_dir / "runtime_attestation.json"
    runtime_attestation = load_json(runtime_path, {})
    runtime_attestation_valid = attestation_is_formal(
        runtime_attestation,
        method=method,
        task=task,
        seed=seed,
    )
    manifest_hashes = manifest.get("hashes", {}) if isinstance(manifest, dict) else {}
    runtime_hash_bound = bool(
        runtime_path.is_file()
        and isinstance(manifest_hashes, dict)
        and isinstance(manifest_hashes.get("runtime_attestation_sha256"), str)
        and len(manifest_hashes["runtime_attestation_sha256"]) == 64
        and manifest_hashes.get("runtime_attestation_sha256") == sha256(runtime_path)
        and isinstance(runtime_attestation.get("code_hashes"), dict)
        and bool(runtime_attestation.get("code_hashes"))
        and manifest_hashes.get("attested_code_hashes") == runtime_attestation.get("code_hashes")
    )
    seed_path = run_dir / "seed_application.json"
    seed_proof = load_json(seed_path, {})
    attested_algorithm = (
        runtime_attestation.get("expected_runtime", {}).get("algorithm_id")
        if isinstance(runtime_attestation.get("expected_runtime"), dict)
        else None
    )
    if method == "radp":
        # Keep the seed proof bound to the algorithm actually attested by the
        # run, but reject unregistered RADP IDs instead of falling through to
        # the PiEvo proof type.
        expected_algorithm = (
            attested_algorithm
            if attested_algorithm in RADP_ALGORITHM_IDS
            else "__invalid_radp_algorithm__"
        )
    else:
        expected_algorithm = "pievo_matched_release_v1"
    seed_attested = seed_proof_valid(seed_proof, seed=seed, algorithm_id=expected_algorithm)
    paths = manifest.get("paths", {}) if isinstance(manifest, dict) else {}
    approval_path_value = paths.get("formal_approval") if isinstance(paths, dict) else None
    approval_path = Path(approval_path_value) if isinstance(approval_path_value, str) and approval_path_value else None
    approval = load_json(approval_path, {}) if approval_path else {}
    approval_valid = bool(
        approval_path
        and isinstance(manifest_hashes, dict)
        and manifest_hashes.get("formal_approval_sha256") == sha256(approval_path)
        and formal_approval_is_valid(
            approval,
            method=method,
            task=task,
            code_hashes=runtime_attestation.get("code_hashes", {}),
            embedding_model=runtime_attestation.get("embedding_model", {}),
        )
    )
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    token_usage = read_cumulative_usage(run_dir / "token_usage.json")
    token_ceiling = runtime.get("token_ceiling") if isinstance(runtime, dict) else None
    token_accounting_valid = bool(
        token_usage["snapshot_index"] is not None
        and int(token_usage["total"] or 0) > 0
        and isinstance(token_ceiling, int)
        and token_ceiling > 0
        and int(token_usage["total"] or 0) <= token_ceiling
    )
    radp_audit_path = run_dir / "radp_runtime_audit.jsonl"
    radp_integration_valid = (
        radp_runtime_audit_valid(radp_audit_path, task=task)
        if method == "radp"
        else not radp_audit_path.exists()
    )
    implementation_invalid_reasons: list[str] = []
    checks = {
        "manifest_schema_v3": manifest_schema_ok,
        "manifest_identity": manifest_identity_ok,
        "runtime_attestation": runtime_attestation_valid,
        "runtime_hash_bound": runtime_hash_bound,
        "formal_approval": approval_valid,
        "seed_application": seed_attested,
        "token_accounting_and_ceiling": token_accounting_valid,
        "radp_main_loop_integration": radp_integration_valid,
    }
    implementation_invalid_reasons.extend(name for name, passed in checks.items() if not passed)
    metrics = trajectory_metrics(task, primary, expected_primary)
    metrics.update(
        {
            "schema_version": "official_four_task_run_metrics.v3",
            "task": task,
            "method": method,
            "seed": seed,
            "run_dir": str(run_dir.resolve()),
            "run_manifest_present": manifest_path.is_file(),
            "run_status": manifest.get("status"),
            "run_track": identity.get("track") if isinstance(identity, dict) else None,
            "manifest_identity_ok": manifest_identity_ok,
            "manifest_schema_ok": manifest_schema_ok,
            "official_task_assets_verified": official_task_assets_verified,
            "matched_evaluator_assets_verified": matched_evaluator_assets_verified,
            "audit_present": audit_path.is_file(),
            "audit_rows": len(rows),
            "invalid_attempts": effective_invalid,
            "duplicate_suppressed": effective_duplicate,
            "verification_calls": len(verification),
            "prefilter_invalid": prefilter_invalid,
            "prefilter_duplicate": prefilter_duplicate,
            "afe_controller_invalid": afe_controller_invalid,
            "afe_controller_duplicate": afe_controller_duplicate,
            "invalid_rate": effective_invalid / (len(primary) + effective_invalid) if primary or effective_invalid else None,
            "duplicate_rate": effective_duplicate / (len(primary) + effective_invalid + effective_duplicate) if primary or effective_invalid or effective_duplicate else None,
            "total_evaluator_calls": len(rows),
            "audit_sha256": sha256(audit_path),
            "manifest_sha256": sha256(manifest_path),
            "token_prompt": token_usage["prompt"],
            "token_completion": token_usage["completion"],
            "token_total": token_usage["total"],
            "token_snapshot_index": token_usage["snapshot_index"],
            "token_ceiling": token_ceiling,
            "token_accounting_valid": token_accounting_valid,
            "runtime_attestation_valid": runtime_attestation_valid,
            "runtime_attestation_sha256": sha256(runtime_path),
            "runtime_hash_bound": runtime_hash_bound,
            "formal_approval_valid": approval_valid,
            "seed_attested": seed_attested,
            "radp_integration_valid": radp_integration_valid,
            "implementation_invalid_reasons": implementation_invalid_reasons,
            "implementation_valid": not implementation_invalid_reasons,
            "legacy_diagnostic_only": not manifest_schema_ok,
            "candidate_count": len({str(row.get("input")) for row in primary}),
            "candidate_repeats": len(primary) - len({str(row.get("input")) for row in primary}),
            "apd_exact_available": False,
        }
    )
    # The evaluator audit is the source of truth for the primary trajectory.
    indices = [int(row.get("call_index", 0)) for row in primary]
    metrics["primary_indices_contiguous"] = indices == list(range(1, len(indices) + 1))
    metrics["formal_run_candidate"] = bool(
        metrics["run_manifest_present"]
        and metrics["audit_present"]
        and metrics["run_status"] == "completed"
        and metrics["manifest_identity_ok"]
        and metrics["implementation_valid"]
        and metrics["matched_evaluator_assets_verified"]
        and metrics["primary_budget_complete"]
        and metrics["primary_indices_contiguous"]
        and metrics["all_values_finite"]
    )
    return metrics


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(values: list[float], repeats: int = 10000, seed: int = 0) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    rng = random.Random(seed)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(repeats)]
    return {"n": len(values), "mean": mean(values), "lower": percentile(samples, 0.025), "upper": percentile(samples, 0.975)}


def paired_sign_permutation(differences: list[float]) -> dict[str, Any]:
    if not differences:
        return {"n": 0, "mean": None, "two_sided_p": None}
    observed = abs(mean(differences))
    extreme = 0
    total = 1 << len(differences)
    for bits in range(total):
        signed = [value if bits & (1 << index) else -value for index, value in enumerate(differences)]
        if abs(mean(signed)) >= observed - 1e-12:
            extreme += 1
    return {"n": len(differences), "mean": mean(differences), "two_sided_p": extreme / total}


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in metrics:
        grouped[(item["method"], item["task"])].append(item)
    task_method: dict[str, Any] = {}
    for (method, task), rows in sorted(grouped.items()):
        sq = [float(row["sq_percent"]) for row in rows if finite_number(row.get("sq_percent"))]
        auoc = [float(row["auoc_percent"]) for row in rows if finite_number(row.get("auoc_percent"))]
        invalid = [float(row["invalid_rate"]) for row in rows if finite_number(row.get("invalid_rate"))]
        duplicate = [float(row["duplicate_rate"]) for row in rows if finite_number(row.get("duplicate_rate"))]
        task_method[f"{method}:{task}"] = {
            "method": method,
            "task": task,
            "run_count": len(rows),
            "seeds": sorted(int(row["seed"]) for row in rows),
            "formal_run_count": sum(bool(row.get("formal_run_candidate")) for row in rows),
            "sq_percent": mean_std(sq),
            "auoc_percent": mean_std(auoc),
            "invalid_rate": mean_std(invalid),
            "duplicate_rate": mean_std(duplicate),
            "total_evaluator_calls": mean_std([float(row["total_evaluator_calls"]) for row in rows]),
        }
    return task_method


def gate_decision(metrics: list[dict[str, Any]], expected_primary: int, reliability_cost_pareto: bool) -> dict[str, Any]:
    eligible_metrics = [item for item in metrics if item.get("formal_run_candidate") is True]
    grouped = aggregate(eligible_metrics)
    diagnostic_grouped = aggregate(metrics)
    implementation_invalid_runs = [
        {
            "method": item.get("method"),
            "task": item.get("task"),
            "seed": item.get("seed"),
            "reasons": item.get("implementation_invalid_reasons", []),
        }
        for item in metrics
        if item.get("implementation_valid") is not True
    ]
    required_keys = {f"{method}:{task}" for method in METHODS for task in TASKS}
    present_keys = set(grouped)
    matrix_complete = required_keys <= present_keys and all(
        grouped[key]["run_count"] == len(SEEDS)
        and grouped[key]["formal_run_count"] == len(SEEDS)
        and grouped[key]["seeds"] == list(SEEDS)
        for key in required_keys & present_keys
    )
    checks: dict[str, Any] = {
        "complete_4_task_2_method_3_run_matrix": matrix_complete,
        "all_selected_runs_implementation_attested": not implementation_invalid_runs,
    }
    reasons: list[str] = []

    control_task_sq: dict[str, float] = {}
    radp_task_sq: dict[str, float] = {}
    control_task_auoc: dict[str, float] = {}
    radp_task_auoc: dict[str, float] = {}
    sq_differences: list[float] = []
    auoc_differences: list[float] = []
    per_task: dict[str, Any] = {}
    invalid_pass = True
    duplicate_pass = True
    for task in TASKS:
        control = grouped.get(f"pievo:{task}", {})
        radp = grouped.get(f"radp:{task}", {})
        c_sq = control.get("sq_percent", {}).get("mean")
        r_sq = radp.get("sq_percent", {}).get("mean")
        c_auoc = control.get("auoc_percent", {}).get("mean")
        r_auoc = radp.get("auoc_percent", {}).get("mean")
        control_task_sq[task] = c_sq
        radp_task_sq[task] = r_sq
        control_task_auoc[task] = c_auoc
        radp_task_auoc[task] = r_auoc
        if c_sq is not None and r_sq is not None:
            sq_differences.append(r_sq - c_sq)
        if c_auoc is not None and r_auoc is not None:
            auoc_differences.append(r_auoc - c_auoc)
        sq_drop = None if c_sq in (None, 0) or r_sq is None else max(0.0, (c_sq - r_sq) / abs(c_sq) * 100.0)
        auoc_drop = None if c_auoc in (None, 0) or r_auoc is None else max(0.0, (c_auoc - r_auoc) / abs(c_auoc) * 100.0)
        c_invalid = control.get("invalid_rate", {}).get("mean")
        r_invalid = radp.get("invalid_rate", {}).get("mean")
        c_duplicate = control.get("duplicate_rate", {}).get("mean")
        r_duplicate = radp.get("duplicate_rate", {}).get("mean")
        invalid_ok = c_invalid is not None and r_invalid is not None and r_invalid <= c_invalid + 1e-12
        duplicate_ok = c_duplicate is not None and r_duplicate is not None and r_duplicate <= c_duplicate + 1e-12
        invalid_pass = invalid_pass and invalid_ok
        duplicate_pass = duplicate_pass and duplicate_ok
        per_task[task] = {
            "control_sq_percent": c_sq,
            "radp_sq_percent": r_sq,
            "sq_relative_drop_percent": sq_drop,
            "sq_noninferiority_pass": sq_drop is not None and sq_drop <= 3.0,
            "control_auoc_percent": c_auoc,
            "radp_auoc_percent": r_auoc,
            "auoc_relative_drop_percent": auoc_drop,
            "auoc_noninferiority_pass": auoc_drop is not None and auoc_drop <= 3.0,
            "invalid_rate_not_higher": invalid_ok,
            "duplicate_rate_not_higher": duplicate_ok,
        }

    # The formal comparison is paired by task and seed.  A task-level mean is
    # useful for reporting, but it must not replace the 12 independent paired
    # cells used by the pre-registered strict-effect gate.
    by_pair = {
        (str(item.get("task")), int(item.get("seed")), str(item.get("method"))): item
        for item in metrics
    }
    paired_cells: list[dict[str, Any]] = []
    for task, seed in itertools.product(TASKS, SEEDS):
        control = by_pair.get((task, seed, "pievo"))
        radp = by_pair.get((task, seed, "radp"))
        if not control or not radp:
            continue
        if not control.get("formal_run_candidate") or not radp.get("formal_run_candidate"):
            continue
        if not finite_number(control.get("sq_percent")) or not finite_number(radp.get("sq_percent")):
            continue
        if not finite_number(control.get("auoc_percent")) or not finite_number(radp.get("auoc_percent")):
            continue
        paired_cells.append(
            {
                "task": task,
                "seed": seed,
                "control_sq_percent": float(control["sq_percent"]),
                "radp_sq_percent": float(radp["sq_percent"]),
                "sq_difference": float(radp["sq_percent"]) - float(control["sq_percent"]),
                "control_auoc_percent": float(control["auoc_percent"]),
                "radp_auoc_percent": float(radp["auoc_percent"]),
                "auoc_difference": float(radp["auoc_percent"]) - float(control["auoc_percent"]),
            }
        )
    paired_sq_differences = [float(cell["sq_difference"]) for cell in paired_cells]
    paired_auoc_differences = [float(cell["auoc_difference"]) for cell in paired_cells]
    paired_sq_bootstrap = bootstrap_mean_interval(paired_sq_differences, repeats=BOOTSTRAP_REPEATS)
    paired_auoc_bootstrap = bootstrap_mean_interval(paired_auoc_differences, repeats=BOOTSTRAP_REPEATS)
    paired_sq_permutation = paired_sign_permutation(paired_sq_differences)
    paired_auoc_permutation = paired_sign_permutation(paired_auoc_differences)
    paired_cells_complete = len(paired_cells) == len(TASKS) * len(SEEDS)
    paired_sq_mean_positive = bool(
        paired_sq_bootstrap["mean"] is not None and paired_sq_bootstrap["mean"] > 0
    )
    paired_sq_interval_pass = bool(
        paired_sq_bootstrap["lower"] is not None and paired_sq_bootstrap["lower"] > 0
    )
    exact_paired_permutation_reported = paired_sq_permutation["n"] == len(TASKS) * len(SEEDS)
    sq_means = [control_task_sq[t] for t in TASKS if control_task_sq[t] is not None]
    radp_sq_means = [radp_task_sq[t] for t in TASKS if radp_task_sq[t] is not None]
    auoc_means = [control_task_auoc[t] for t in TASKS if control_task_auoc[t] is not None]
    radp_auoc_means = [radp_task_auoc[t] for t in TASKS if radp_task_auoc[t] is not None]
    aggregate_control_sq = mean(sq_means) if len(sq_means) == len(TASKS) else None
    aggregate_radp_sq = mean(radp_sq_means) if len(radp_sq_means) == len(TASKS) else None
    aggregate_control_auoc = mean(auoc_means) if len(auoc_means) == len(TASKS) else None
    aggregate_radp_auoc = mean(radp_auoc_means) if len(radp_auoc_means) == len(TASKS) else None
    aggregate_sq_pass = aggregate_radp_sq is not None and aggregate_control_sq is not None and aggregate_radp_sq > aggregate_control_sq
    aggregate_auoc_pass = aggregate_radp_auoc is not None and aggregate_control_auoc is not None and aggregate_radp_auoc >= aggregate_control_auoc * 0.97
    sq_noninferiority = all(item["sq_noninferiority_pass"] for item in per_task.values()) and len(per_task) == len(TASKS)
    auoc_noninferiority = all(item["auoc_noninferiority_pass"] for item in per_task.values()) and len(per_task) == len(TASKS)
    checks.update(
        {
            "aggregate_sq_strict_improvement": aggregate_sq_pass,
            "per_task_sq_noninferiority_3_percent": sq_noninferiority,
            "aggregate_auoc_noninferiority": aggregate_auoc_pass,
            "per_task_auoc_noninferiority_3_percent": auoc_noninferiority,
            "invalid_rate_not_higher_than_control": invalid_pass,
            "duplicate_rate_not_higher_than_control": duplicate_pass,
            "reliability_discovery_cost_pareto": reliability_cost_pareto,
            "complete_12_paired_task_seed_cells": paired_cells_complete,
            "paired_sq_mean_difference_positive": paired_sq_mean_positive,
            "paired_sq_bootstrap_95_lower_above_zero": paired_sq_interval_pass,
            "exact_paired_permutation_reported": exact_paired_permutation_reported,
        }
    )
    required_check_names = (
        "complete_4_task_2_method_3_run_matrix",
        "all_selected_runs_implementation_attested",
        "aggregate_sq_strict_improvement",
        "aggregate_auoc_noninferiority",
        "invalid_rate_not_higher_than_control",
        "duplicate_rate_not_higher_than_control",
        "reliability_discovery_cost_pareto",
        "complete_12_paired_task_seed_cells",
        "paired_sq_mean_difference_positive",
        "paired_sq_bootstrap_95_lower_above_zero",
        "exact_paired_permutation_reported",
    )
    reasons.extend(name for name in required_check_names if not checks[name])
    required_gate_pass = all(checks[name] for name in required_check_names)
    clean_check_names = (
        "complete_4_task_2_method_3_run_matrix",
        "aggregate_sq_strict_improvement",
        "aggregate_auoc_noninferiority",
        "invalid_rate_not_higher_than_control",
        "duplicate_rate_not_higher_than_control",
    )
    reliability_check_names = ("reliability_discovery_cost_pareto",)
    statistics_check_names = (
        "complete_12_paired_task_seed_cells",
        "paired_sq_mean_difference_positive",
        "paired_sq_bootstrap_95_lower_above_zero",
        "exact_paired_permutation_reported",
    )
    gate_levels = {
        "G1_official_matrix_completeness": {
            "pass": bool(
                checks["complete_4_task_2_method_3_run_matrix"]
                and checks["all_selected_runs_implementation_attested"]
            ),
            "checks": [
                "complete_4_task_2_method_3_run_matrix",
                "all_selected_runs_implementation_attested",
            ],
        },
        "G2_clean": {
            "pass": all(bool(checks[name]) for name in clean_check_names),
            "checks": list(clean_check_names),
        },
        "G2_reliability": {
            "pass": all(bool(checks[name]) for name in reliability_check_names),
            "checks": list(reliability_check_names),
        },
        "G2_statistics": {
            "pass": all(bool(checks[name]) for name in statistics_check_names),
            "checks": list(statistics_check_names),
        },
        "external_unlock_from_metrics": {
            "pass": required_gate_pass,
            "requires": ["G1_official_matrix_completeness", "G2_clean", "G2_reliability", "G2_statistics"],
            "note": "Combine with the separate G0 official-readiness preflight before launching external runs.",
        },
    }
    return {
        "schema_version": "official_four_task_gate_decision.v3",
        "expected_primary": expected_primary,
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "runs_per_task_method": len(SEEDS),
        "formal_primary_total": len(TASKS) * len(METHODS) * len(SEEDS) * expected_primary,
        "aggregate": {
            "control_sq_percent": aggregate_control_sq,
            "radp_sq_percent": aggregate_radp_sq,
            "control_auoc_percent": aggregate_control_auoc,
            "radp_auoc_percent": aggregate_radp_auoc,
            "sq_differences_by_task": sq_differences,
            "auoc_differences_by_task": auoc_differences,
            "sq_differences_by_paired_cell": paired_cells,
            "sq_difference_bootstrap": paired_sq_bootstrap,
            "auoc_difference_bootstrap": paired_auoc_bootstrap,
            "sq_difference_permutation": paired_sq_permutation,
            "auoc_difference_permutation": paired_auoc_permutation,
        },
        "diagnostic_aggregate_including_ineligible_runs": diagnostic_grouped,
        "implementation_invalid_runs": implementation_invalid_runs,
        "per_task": per_task,
        "checks": checks,
        "reporting_only_checks": [
            "per_task_sq_noninferiority_3_percent",
            "per_task_auoc_noninferiority_3_percent",
        ],
        "gate_levels": gate_levels,
        "gate_pass": required_gate_pass,
        "strong_statistical_superiority": all(bool(checks[name]) for name in statistics_check_names),
        "apd_claimable": False,
        "reliability_cost_pareto_input": reliability_cost_pareto,
        "external_full_run_unlocked": required_gate_pass,
        "reasons": reasons,
        "claim_boundary": {
            "matched_claim": "strict superiority available only after clean, reliability, and paired-statistics gates pass; otherwise directional matched result only",
            "paper_qwen32b_claim": "forbidden in this Qwen3-235B track",
            "apd": "non-claimable without exact task-specific evaluator",
            "external": "locked unless G0 readiness and metric G1/G2 clean/reliability/statistics gates all pass",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/official_gate_decision.json"))
    parser.add_argument("--expected-primary", type=int, default=24)
    parser.add_argument("--reliability-cost-json", type=Path, default=None)
    parser.add_argument(
        "--run-map-json",
        type=Path,
        default=None,
        help="Optional JSON map keyed as method:task:seed to explicit run directories.",
    )
    args = parser.parse_args()

    run_map = load_json(args.run_map_json, {}) if args.run_map_json else {}
    metrics: list[dict[str, Any]] = []
    for method, task, seed in itertools.product(METHODS, TASKS, SEEDS):
        map_key = f"{method}:{task}:{seed}"
        mapped_dir = run_map.get(map_key) if isinstance(run_map, dict) else None
        run_dir = (
            Path(mapped_dir)
            if isinstance(mapped_dir, str) and mapped_dir
            else args.results_root / f"matched_qwen235b_{method}_{task.lower()}_seed{seed}_formal24"
        )
        metrics.append(run_metrics(task, method, seed, run_dir, args.expected_primary))
    reliability_cost = False
    reliability_detail: dict[str, Any] = {"provided": False, "reason": "no independent equal-total reliability-cost manifest supplied"}
    if args.reliability_cost_json and args.reliability_cost_json.is_file():
        value = load_json(args.reliability_cost_json, {})
        reliability_cost = value.get("pareto_front", False) is True
        reliability_detail = {"provided": True, "path": str(args.reliability_cost_json.resolve()), "value": value}
    decision = gate_decision(metrics, args.expected_primary, reliability_cost)
    decision["reliability_cost_detail"] = reliability_detail
    decision["run_metrics"] = metrics
    decision["generated_from"] = {
        "results_root": str(args.results_root.resolve()),
        "run_map_json": str(args.run_map_json.resolve()) if args.run_map_json else None,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    if metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in metrics for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics)
    print(json.dumps({
        "output": str(args.output),
        "gate_pass": decision["gate_pass"],
        "external_full_run_unlocked": decision["external_full_run_unlocked"],
        "matrix_complete": decision["checks"]["complete_4_task_2_method_3_run_matrix"],
        "reasons": decision["reasons"],
    }, ensure_ascii=False, indent=2))
    return 0 if decision["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
