#!/usr/bin/env python3
"""Audit one paired official-MBO development screen.

The source-confusion mode is an evaluator-side diagnostic: both methods see
the same preregistered primary spike, while RADP can compare it with the clean
released surrogate verifier.  This is not a formal benchmark or an
independent physical verification channel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from experiments.formal_usage import read_cumulative_usage
    from experiments.formal_runtime_contract import (
        repository_hashes,
        sha256 as file_sha256,
    )
except ImportError:
    from formal_usage import read_cumulative_usage
    from formal_runtime_contract import repository_hashes, sha256 as file_sha256


AUDIT_NAME = "official_chembl_tool_audit.jsonl"
GATE_NAME = "radp_source_gate_audit.jsonl"
RUNTIME_NAME = "radp_runtime_audit.jsonl"
PREFILTER_NAME = "candidate_prefilter_audit.jsonl"
AFE_PROPOSAL_NAME = "radp_afe_proposal_audit.jsonl"
CONTEXT_BOUNDARY_NAME = "agent_context_boundary_audit.jsonl"
PORTFOLIO_NAME = "radp_candidate_portfolio_audit.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_value(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    value = read_value(path)
    return value if isinstance(value, dict) else {}


def current_attested_hashes(
    *,
    attested_hashes: Any,
    manifest_paths: dict[str, Any],
    official_repo: Path,
    method: str,
) -> dict[str, str | None]:
    """Recompute exactly the code/input hashes present in the attestation.

    Runtime attestations include both worktree code hashes and hashes of the
    task/model/protocol inputs.  Comparing the complete attestation with only
    repository hashes makes every fresh run look stale.  The input paths are
    resolved from the immutable run manifest, and a missing path fails closed.
    Older diagnostic fixtures without input hashes retain the repository-only
    comparison for backwards-compatible auditing.
    """

    if not isinstance(attested_hashes, dict):
        return {}
    current: dict[str, str | None] = repository_hashes(official_repo, method)
    input_path_names = {
        "input_task_config": "task_config",
        "input_model_config": "model_config",
        "input_protocol": "protocol",
    }
    for hash_name, path_name in input_path_names.items():
        if hash_name not in attested_hashes:
            continue
        raw_path = manifest_paths.get(path_name)
        path = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
        current[hash_name] = file_sha256(path) if path is not None else None
    return current


def primary_rows(directory: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(directory / AUDIT_NAME)
    return [
        row
        for row in rows
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
        and math.isfinite(float(row["output"]))
    ]


def audit_method(
    directory: Path,
    *,
    method: str,
    seed: int,
    budget: int,
    mode: str,
    token_ceiling: int,
) -> dict[str, Any]:
    audit_rows = read_jsonl(directory / AUDIT_NAME)
    primaries = primary_rows(directory)
    corruption_rows = [
        row
        for row in audit_rows
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("corruption_mode") is not None
    ]
    corruption_calls = sorted(
        int(row["call_index"])
        for row in corruption_rows
        if row.get("corruption_applied") is True
        and isinstance(row.get("call_index"), int)
    )
    invalid = sum(
        1
        for row in audit_rows
        if row.get("tool_name") == "evaluate_chembl" and row.get("success") is not True
    )
    duplicates = sum(
        1
        for row in audit_rows
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("duplicate_suppressed") is True
    )
    prefilter_rows = read_jsonl(directory / PREFILTER_NAME)
    prefilter_invalid = sum(
        row.get("disposition") == "invalid" for row in prefilter_rows
    )
    prefilter_duplicates = sum(
        row.get("disposition") == "canonical_duplicate" for row in prefilter_rows
    )
    afe_proposal_rows = read_jsonl(directory / AFE_PROPOSAL_NAME)
    afe_controller_invalid = sum(
        row.get("disposition") == "afe_invalid_proposal"
        for row in afe_proposal_rows
    )
    afe_controller_duplicates = sum(
        row.get("disposition") == "afe_canonical_duplicate"
        for row in afe_proposal_rows
    )
    afe_rejections_pre_evaluator = all(
        row.get("entered_evaluator") is False for row in afe_proposal_rows
    )
    manifest = read_json(directory / "run_manifest.json")
    run_identity = manifest.get("run_identity", {})
    if not isinstance(run_identity, dict):
        run_identity = {}
    final_evidence = manifest.get("final_evidence", {})
    if not isinstance(final_evidence, dict):
        final_evidence = {}
    manifest_runtime = manifest.get("runtime", {})
    if not isinstance(manifest_runtime, dict):
        manifest_runtime = {}
    manifest_paths = manifest.get("paths", {})
    if not isinstance(manifest_paths, dict):
        manifest_paths = {}
    runtime_attestation = read_json(directory / "runtime_attestation.json")
    attested_hashes = runtime_attestation.get("code_hashes", {})
    official_repo = Path(str(manifest_paths.get("official_repo", "")))
    current_hashes = (
        current_attested_hashes(
            attested_hashes=attested_hashes,
            manifest_paths=manifest_paths,
            official_repo=official_repo,
            method=method,
        )
        if official_repo.is_dir()
        else {}
    )
    runtime_code_hashes_current = bool(attested_hashes) and bool(current_hashes) and (
        dict(attested_hashes) == current_hashes
    )
    token = read_cumulative_usage(directory / "token_usage.json")
    runtime_rows = read_jsonl(directory / RUNTIME_NAME)
    gate_rows = read_jsonl(directory / GATE_NAME)
    context_boundary_rows = read_jsonl(directory / CONTEXT_BOUNDARY_NAME)
    portfolio_rows = read_jsonl(directory / PORTFOLIO_NAME)
    portfolio_selection_rows = [
        row
        for row in portfolio_rows
        if row.get("schema_version")
        in {"radp_candidate_portfolio_audit.v1", "radp_candidate_portfolio_audit.v2"}
    ]
    principle_boundary_rows = [
        row
        for row in portfolio_rows
        if row.get("schema_version") == "radp_principle_guidance_boundary.v1"
    ]
    afe_schema_rows = {
        str(row.get("schema_version"))
        for row in portfolio_rows
        if row.get("schema_version") in {
            "radp_afe_portfolio.v1",
            "radp_afe_acquisition.v1",
            "radp_afe_portfolio.v3",
            "radp_afe_acquisition.v3",
            "radp_afe_portfolio.v8",
            "radp_afe_acquisition.v8",
        }
    }
    context_boundary_roles = sorted(
        {
            role
            for row in context_boundary_rows
            for role in ("principle", "hypothesis", "experiment")
            if role in str(row.get("agent", "")).strip().lower()
        }
    )
    context_boundary_all_blocked = bool(context_boundary_rows) and all(
        row.get("event") == "guidance_only_context_reset"
        and row.get("admitted_count") == 0
        and row.get("blocked_count") == row.get("incoming_count")
        for row in context_boundary_rows
    )
    primary_inputs = {
        str(row.get("input", "")).strip()
        for row in primaries
        if str(row.get("input", "")).strip()
    }
    terminal_gate_rows = [
        row for row in gate_rows if row.get("gate_terminal") is True
    ]
    terminal_gate_candidates = {
        str(row.get("hypothesis", "")).strip()
        for row in terminal_gate_rows
        if str(row.get("hypothesis", "")).strip()
    }
    rejected_gate_rows = [
        row
        for row in gate_rows
        if row.get("raw_update_allowed") is False
        or row.get("update_allowed") is False
        or row.get("source_decision") in {"replace", "quarantine", "stop"}
    ]
    repaired_gate_rows = [
        row
        for row in rejected_gate_rows
        if isinstance(row.get("primary_observation"), (int, float))
        and not isinstance(row.get("primary_observation"), bool)
        and isinstance(row.get("verified_observation"), (int, float))
        and not isinstance(row.get("verified_observation"), bool)
        and math.isfinite(float(row["primary_observation"]))
        and math.isfinite(float(row["verified_observation"]))
        and not math.isclose(
            float(row["primary_observation"]),
            float(row["verified_observation"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ]
    runtime_started = any(
        row.get("event") == "runtime_start"
        and row.get("algorithm_id")
        in {
            "radp_search_active_v1",
            "radp_search_afe_v1",
            "radp_search_afe_v8",
        }
        and row.get("mode") == "active"
        and row.get("provider") == "official_mbo"
        for row in runtime_rows
    )
    portfolio_strategy = str(manifest_runtime.get("radp_search_strategy", "legacy_v1"))
    afe_schema_version = "v8" if portfolio_strategy == "afe_v8" else "v3"
    expected_roles = [
        "local_exploitation",
        "cross_scaffold",
        "orthogonal_exploration",
    ]
    def valid_portfolio_row(row: dict[str, Any]) -> bool:
        try:
            selection_index = int(row.get("selected_role_index"))
        except (TypeError, ValueError):
            return False
        if row.get("strategy_version") != "portfolio_v2":
            return False
        if row.get("provisional_principles_used_for_selection") is not False:
            return False
        if row.get("source_gate_values_used_for_selection") is not False:
            return False
        if row.get("verifier_values_used_for_selection") is not False:
            return False
        if row.get("schema_version") == "radp_candidate_portfolio_audit.v2" and not isinstance(
            row.get("seed_bank_prioritized"), bool
        ):
            return False

        # v1 rows have no explicit fallback metadata. Preserve their original
        # strict contract for compatibility with old diagnostics.
        if row.get("schema_version") == "radp_candidate_portfolio_audit.v1":
            return (
                row.get("target_role")
                == expected_roles[selection_index % len(expected_roles)]
                and row.get("target_role_available") is True
            )

        try:
            requested_index = int(row.get("requested_role_index"))
            effective_index = row.get("effective_role_index")
            effective_index = (
                int(effective_index) if effective_index is not None else None
            )
        except (TypeError, ValueError):
            return False
        if not 0 <= requested_index < len(expected_roles):
            return False
        requested_role = expected_roles[requested_index]
        if row.get("target_role") != requested_role:
            return False
        selected_role = row.get("selected_role")
        role_map = row.get("portfolio_roles")
        if not isinstance(role_map, dict):
            return False
        available_roles = {
            role for role in expected_roles if role in set(role_map.values())
        }
        target_available = row.get("target_role_available") is True
        if (requested_role in available_roles) != target_available:
            return False
        if target_available:
            return (
                row.get("fallback_used") is False
                and selected_role == requested_role
                and effective_index == requested_index
                and row.get("fallback_from_role") is None
                and row.get("fallback_to_role") is None
                and row.get("fallback_reason") is None
            )

        if row.get("fallback_used") is not True:
            return False
        if row.get("fallback_from_role") != requested_role:
            return False
        if row.get("fallback_to_role") != selected_role:
            return False
        if row.get("fallback_reason") != "requested_role_unavailable":
            return False
        if selected_role not in available_roles or effective_index is None:
            return False
        expected_effective_index = next(
            (
                (requested_index + offset) % len(expected_roles)
                for offset in range(1, len(expected_roles) + 1)
                if expected_roles[(requested_index + offset) % len(expected_roles)]
                in available_roles
            ),
            None,
        )
        return (
            expected_effective_index == effective_index
            and selected_role == expected_roles[effective_index]
        )

    portfolio_roles_valid = all(valid_portfolio_row(row) for row in portfolio_selection_rows)
    principle_boundary_valid = all(
        row.get("proposal_only") is True
        and (
            row.get("all_guidance_principles_are_accepted") is True
            or (
                row.get("bootstrap_prior_context") is True
                and row.get("guidance_principles_are_outcome_blind") is True
            )
        )
        and row.get("source_gate_payload_exposed") is False
        and row.get("verifier_payload_exposed") is False
        for row in principle_boundary_rows
    )
    best_observed = max((float(row["output"]) for row in primaries), default=None)
    true_outputs = [
        float(row.get("raw_output", row["output"]))
        for row in primaries
        if isinstance(row.get("raw_output", row["output"]), (int, float))
        and not isinstance(row.get("raw_output", row["output"]), bool)
        and math.isfinite(float(row.get("raw_output", row["output"])))
    ]
    best_true = max(true_outputs, default=None)
    return {
        "method": method,
        "seed": seed,
        "directory": str(directory.resolve()),
        "manifest_status": manifest.get("status"),
        "manifest_identity": {
            "task": run_identity.get("task"),
            "method": run_identity.get("method"),
            "seed": run_identity.get("seed"),
            "primary_budget": run_identity.get("primary_budget"),
        },
        "seed_application_passed": final_evidence.get("seed_application_passed") is True,
        "runtime_code_hashes_current": runtime_code_hashes_current,
        "primary_count": len(primaries),
        "invalid_count": invalid,
        "duplicate_count": duplicates,
        "prefilter_invalid_count": prefilter_invalid,
        "prefilter_duplicate_count": prefilter_duplicates,
        "afe_controller_audit_count": len(afe_proposal_rows),
        "afe_controller_invalid_count": afe_controller_invalid,
        "afe_controller_duplicate_count": afe_controller_duplicates,
        "afe_controller_rejections_pre_evaluator": afe_rejections_pre_evaluator,
        "effective_invalid_count": invalid + prefilter_invalid,
        "effective_duplicate_count": duplicates + prefilter_duplicates,
        "candidate_prefilter_enabled": (
            manifest_runtime.get("candidate_prefilter_enabled") is True
        ),
        "candidate_prefilter_rejection_limit": manifest_runtime.get(
            "candidate_prefilter_rejection_limit"
        ),
        "candidate_prefilter_empty_pool_limit": manifest_runtime.get(
            "candidate_prefilter_empty_pool_limit"
        ),
        "guidance_only_context": (
            manifest_runtime.get("guidance_only_context") is True
        ),
        "context_boundary_count": len(context_boundary_rows),
        "context_boundary_roles": context_boundary_roles,
        "context_boundary_all_incoming_blocked": context_boundary_all_blocked,
        "corruption_calls": corruption_calls,
        "best_observed_output": best_observed,
        "best_true_output": best_true,
        "token_usage": token,
        "token_ceiling_passed": (
            token["snapshot_index"] is not None
            and int(token["total"] or 0) <= token_ceiling
        ),
        "runtime_started": runtime_started,
        "gate_count": len(gate_rows),
        "terminal_gate_count": len(terminal_gate_rows),
        "terminal_gate_candidate_count": len(terminal_gate_candidates),
        "terminal_gate_candidates_match_primary": (
            terminal_gate_candidates == primary_inputs
        ),
        "gate_finalized": any(
            row.get("event") == "gate_finalized" for row in runtime_rows
        ),
        "gate_rejection_count": len(rejected_gate_rows),
        "gate_repair_count": len(repaired_gate_rows),
        "principle_transition_count": sum(
            row.get("event") == "principle_transition" for row in runtime_rows
        ),
        "radp_search_strategy": portfolio_strategy,
        "afe_schema_version": afe_schema_version,
        "portfolio_audit_count": len(portfolio_selection_rows),
        "portfolio_roles_valid": portfolio_roles_valid,
        "portfolio_seed_bank_count": sum(
            row.get("seed_bank_prioritized") is True
            for row in portfolio_selection_rows
        ),
        "afe_portfolio_audit_count": sum(
            row.get("schema_version") == f"radp_afe_portfolio.{afe_schema_version}"
            for row in portfolio_rows
        ),
        "afe_acquisition_audit_count": sum(
            row.get("schema_version") == f"radp_afe_acquisition.{afe_schema_version}"
            for row in portfolio_rows
        ),
        "afe_audit_schemas": sorted(afe_schema_rows),
        "principle_guidance_boundary_count": len(principle_boundary_rows),
        "principle_guidance_boundary_valid": principle_boundary_valid,
    }


def audit(
    root: Path,
    *,
    seed: int,
    budget: int,
    mode: str,
    token_ceiling: int,
) -> dict[str, Any]:
    methods = {
        method: audit_method(
            root / method,
            method=method,
            seed=seed,
            budget=budget,
            mode=mode,
            token_ceiling=token_ceiling,
        )
        for method in ("pievo", "radp")
    }
    pievo = methods["pievo"]
    radp = methods["radp"]
    expected_corruption = [2, 3] if mode == "source_confusion" else []
    checks: dict[str, bool] = {
        "both_methods_present": all(Path(row["directory"]).is_dir() for row in methods.values()),
        "manifests_completed": all(row["manifest_status"] == "completed" for row in methods.values()),
        "manifest_identities_match_protocol": all(
            row["manifest_identity"]
            == {
                "task": "MBO",
                "method": method,
                "seed": seed,
                "primary_budget": budget,
            }
            for method, row in methods.items()
        ),
        "seed_application_passed": all(
            row["seed_application_passed"] for row in methods.values()
        ),
        "runtime_code_hashes_current": all(
            row["runtime_code_hashes_current"] for row in methods.values()
        ),
        "exact_primary_budget": all(row["primary_count"] == budget for row in methods.values()),
        "same_corruption_schedule": all(row["corruption_calls"] == expected_corruption for row in methods.values()),
        "token_ceiling_passed": all(bool(row["token_ceiling_passed"]) for row in methods.values()),
        "matched_candidate_prefilter": all(
            row["candidate_prefilter_enabled"] is True
            and row["candidate_prefilter_rejection_limit"] == 12
            and row["candidate_prefilter_empty_pool_limit"] == 2
            for row in methods.values()
        ),
        "invalid_duplicate_noninferior": (
            radp["effective_invalid_count"] + radp["effective_duplicate_count"]
            <= pievo["effective_invalid_count"] + pievo["effective_duplicate_count"]
        ),
        "radp_portfolio_audit_if_enabled": (
            radp["radp_search_strategy"] != "portfolio_v2"
            or (
                radp["portfolio_audit_count"] > 0
                and radp["portfolio_roles_valid"]
                and radp["principle_guidance_boundary_count"] > 0
                and radp["principle_guidance_boundary_valid"]
            )
        ),
        "radp_portfolio_seed_bank_if_enabled": (
            radp["radp_search_strategy"] != "portfolio_v2"
            or radp["portfolio_seed_bank_count"] >= 3
        ),
        "radp_afe_audit_if_enabled": (
            radp["radp_search_strategy"] != "afe_v8"
            or (
                radp["afe_portfolio_audit_count"] > 0
                and radp["afe_acquisition_audit_count"] >= budget
                and radp["afe_audit_schemas"]
                == sorted(
                    [
                        f"radp_afe_acquisition.{radp['afe_schema_version']}",
                        f"radp_afe_portfolio.{radp['afe_schema_version']}",
                    ]
                )
            )
        ),
        "radp_afe_rejections_pre_evaluator": (
            radp["radp_search_strategy"] != "afe_v8"
            or radp["afe_controller_rejections_pre_evaluator"]
        ),
    }
    if mode == "source_confusion":
        checks.update(
            {
                "radp_runtime_started": bool(radp["runtime_started"]),
                "radp_guidance_only_context_enabled": bool(
                    radp["guidance_only_context"]
                ),
                "radp_context_boundary_audited": (
                    radp["context_boundary_roles"]
                    == ["experiment", "hypothesis", "principle"]
                    and radp["context_boundary_all_incoming_blocked"] is True
                ),
                "radp_gate_present": radp["gate_count"] > 0,
                "radp_gate_finalized": bool(radp["gate_finalized"]),
                "radp_terminal_gate_count": (
                    radp["terminal_gate_count"] == budget
                ),
                "radp_terminal_gate_candidates_match_primary": bool(
                    radp["terminal_gate_candidates_match_primary"]
                ),
                "radp_rejects_both_injected_source_failures": radp["gate_rejection_count"] >= 2,
                "radp_repairs_both_injected_source_failures": radp["gate_repair_count"] >= 2,
                "radp_best_true_output_noninferior": (
                    pievo["best_true_output"] is not None
                    and radp["best_true_output"] is not None
                    and radp["best_true_output"]
                    >= pievo["best_true_output"] * 0.97
                ),
            }
        )
    else:
        checks["clean_mode_has_no_corruption"] = all(
            row["corruption_calls"] == [] for row in methods.values()
        )
    result = {
        "schema_version": "official_mbo_development_screen_audit.v3",
        "task": "MBO",
        "seed": seed,
        "budget": budget,
        "mode": mode,
        "token_ceiling": token_ceiling,
        "methods": methods,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": (
            "Controlled official-MBO development diagnostic only; no formal, paper-setting, "
            "or external benchmark claim is allowed."
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--mode", choices=["clean", "source_confusion"], required=True)
    parser.add_argument("--token-ceiling", type=int, default=75000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.root,
        seed=args.seed,
        budget=args.budget,
        mode=args.mode,
        token_ceiling=args.token_ceiling,
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 12


if __name__ == "__main__":
    raise SystemExit(main())
