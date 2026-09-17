#!/usr/bin/env python3
"""Fail-closed zero-API gates for the RADP-Search SOTA workflow.

This module only inspects local protocol, provenance, manifests, and source
files. It never reads an API key and never starts an evaluator or LLM client.
The gate is intentionally conservative: a missing or stale claim boundary is
reported as a stop, not inferred as ready from a nearby artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
READINESS = ROOT / "experiments/results/official_four_task_readiness_v8.json"
PREFLIGHT = ROOT / "experiments/results/official_formal_preflight_afe_v8.json"
FREEZE = ROOT / "experiments/results/matched_qwen235b_protocol_freeze_afe_v8.json"
G0 = ROOT / "experiments/results/g0_evidence_audit_v8.json"
G1 = ROOT / "experiments/results/g1_fairness_audit_v8.json"
RADP_VERIFIER_READINESS = ROOT / "experiments/results/radp_mbo_independent_verifier_readiness.json"
POOL = ROOT / "experiments/results/official_chembl35_pool_seed0/chembl35_pool_seed0.jsonl"
POOL_SUMMARY = ROOT / "experiments/results/official_chembl35_pool_seed0/chembl35_pool_seed0_summary.json"
CAPABILITIES = ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/radp/capabilities.json"
RUNNER = ROOT / "experiments/run_matched_qwen235b_task.sh"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def freeze_matches(
    protocol: dict[str, Any],
    readiness: dict[str, Any],
    preflight: dict[str, Any],
    freeze: dict[str, Any],
) -> bool:
    """Require the G1 freeze to bind the current protocol and G0 artifacts."""

    repetition = protocol.get("official_repetition_policy", {})
    scope = freeze.get("scope", {})
    return bool(
        freeze.get("schema_version") == "matched_qwen235b_protocol_freeze.v2"
        and freeze.get("protocol_sha256") == sha256(PROTOCOL)
        and freeze.get("readiness_sha256") == sha256(READINESS)
        and freeze.get("preflight_sha256") == sha256(PREFLIGHT)
        and freeze.get("g0_evidence_sha256") == sha256(G0)
        and freeze.get("g1_fairness_sha256") == sha256(G1)
        and scope.get("model") == protocol.get("model")
        and scope.get("model_track") == protocol.get("model_track")
        and scope.get("tasks") == repetition.get("pievo_tasks")
        and scope.get("formal_seeds") == protocol.get("seed_policy", {}).get("formal")
        and scope.get("primary_budget") == protocol.get("primary_budget")
        and scope.get("formal_primary_total") == repetition.get("formal_primary_total")
        and readiness.get("formal_matched_run_ready") is True
        and preflight.get("formal_matched_run_ready") is True
        and load_json(G0).get("passed") is True
        and load_json(G1).get("passed") is True
    )


def protocol_checks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    repetition = protocol.get("official_repetition_policy", {})
    statistical = protocol.get("statistical_policy", {})
    staged = protocol.get("staged_execution", {})
    execution = protocol.get("execution_safety", {})
    return [
        check(
            "model_track",
            protocol.get("model") == "qwen3-235b-a22b-instruct-2507"
            and protocol.get("model_track") == "matched_qwen235b_only",
            {"model": protocol.get("model"), "track": protocol.get("model_track")},
        ),
        check(
            "formal_matrix",
            repetition.get("pievo_tasks") == ["NHO", "MBO", "SPO", "TMC"]
            and repetition.get("pievo_independent_runs") == 3
            and repetition.get("official_primary_per_run") == 24
            and repetition.get("formal_primary_total") == 576,
            {
                "tasks": repetition.get("pievo_tasks"),
                "runs": repetition.get("pievo_independent_runs"),
                "primary_per_run": repetition.get("official_primary_per_run"),
                "formal_primary_total": repetition.get("formal_primary_total"),
            },
        ),
        check(
            "strict_statistics",
            statistical.get("primary_endpoint") == "mean_task_normalized_sq"
            and statistical.get("paired_unit") == "task_seed"
            and statistical.get("paired_cell_count") == 12
            and statistical.get("permutation") == "exact_paired_sign_permutation_two_sided"
            and statistical.get("no_post_hoc_endpoint_or_cell_deletion") is True,
            statistical,
        ),
        check(
            "staged_execution",
            staged.get("evaluator_g0_must_precede_matched_formal") is True
            and staged.get("official_phase_must_precede_external_full") is True
            and staged.get("external_full_run_requires_gate_artifact") is True,
            staged,
        ),
        check(
            "runtime_fail_closed",
            execution.get("formal_runner_requires_g0_preflight") is True
            and execution.get("matched_formal_requires_evaluator_g0") is True
            and execution.get("external_runner_requires_gate_decision_and_g0") is True
            and execution.get("fresh_output_directory_required") is True,
            execution,
        ),
        check(
            "development_budget_semantics",
            protocol.get("g1_fairness_protocol", {})
            .get("budget_accounting", {})
            .get("primary_budget_equal") is True
            and protocol.get("g1_fairness_protocol", {})
            .get("budget_accounting", {})
            .get("verification_cost_recorded_separately") is True
            and protocol.get("g1_fairness_protocol", {})
            .get("budget_accounting", {})
            .get("equal_total_action_budget_required_for_cost_sweep") is True,
            {
                "budget_accounting": protocol.get("g1_fairness_protocol", {}).get("budget_accounting", {}),
            },
        ),
    ]


def evidence_checks(readiness: dict[str, Any], preflight: dict[str, Any]) -> list[dict[str, Any]]:
    summary = load_json(POOL_SUMMARY)
    rows = pool_rows(POOL)
    declared = summary.get("sample_size")
    mbo = readiness.get("tasks", {}).get("MBO", {})
    capability = load_json(CAPABILITIES)
    mbo_capability = capability.get("tasks", {}).get("MBO", {})
    freeze = load_json(FREEZE)
    verifier_readiness = load_json(RADP_VERIFIER_READINESS)
    runner_check = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True, check=False
    )
    return [
        check(
            "readiness_schema",
            readiness.get("schema_version") == "official_four_task_readiness.v8",
            readiness.get("schema_version"),
        ),
        check(
            "mbo_reconstruction_provenance",
            rows == 50000
            and declared == rows
            and mbo.get("controlled_local_pool", {}).get("row_count") == rows
            and mbo.get("hidden_50000_sample_recovered") is False
            and readiness.get("formal_official_run_ready") is False,
            {
                "rows": rows,
                "declared_sample_size": declared,
                "readiness_pool": mbo.get("controlled_local_pool"),
                "formal_official_run_ready": readiness.get("formal_official_run_ready"),
            },
        ),
        check(
            "released_matched_preflight",
            preflight.get("decision") == "READY_FOR_MATCHED_FORMAL"
            and preflight.get("formal_matched_run_ready") is True
            and preflight.get("formal_official_run_ready") is False,
            {
                "decision": preflight.get("decision"),
                "formal_matched_run_ready": preflight.get("formal_matched_run_ready"),
                "formal_official_run_ready": preflight.get("formal_official_run_ready"),
            },
        ),
        check(
            "runner_syntax",
            runner_check.returncode == 0,
            runner_check.stderr.strip() or "bash -n passed",
        ),
        check(
            "radp_four_task_formal_provider_is_not_implicitly_ready",
            capability.get("formal_ready") is False
            and all(
                isinstance(value, dict)
                and value.get("formal_ready") is False
                for name, value in capability.get("tasks", {}).items()
                if name != "MBO"
            ),
            {
                "formal_ready": capability.get("formal_ready"),
                "tasks": capability.get("tasks"),
            },
        ),
        check(
            "g1_protocol_freeze_is_current",
            freeze_matches(protocol=load_json(PROTOCOL), readiness=readiness, preflight=preflight, freeze=freeze),
            {
                "artifact": str(FREEZE),
                "artifact_present": FREEZE.is_file(),
                "schema_version": freeze.get("schema_version"),
                "protocol_sha256": freeze.get("protocol_sha256"),
                "readiness_sha256": freeze.get("readiness_sha256"),
                "preflight_sha256": freeze.get("preflight_sha256"),
                "g0_evidence_sha256": freeze.get("g0_evidence_sha256"),
                "g1_fairness_sha256": freeze.get("g1_fairness_sha256"),
            },
        ),
        check(
            "radp_mbo_verifier_state_is_fail_closed_and_consistent",
            verifier_readiness.get("schema_version")
            == "radp_mbo_independent_verifier_readiness.v2"
            and (
                (
                    mbo_capability.get("formal_ready") is True
                    and mbo_capability.get("independent_channel") is True
                    and mbo_capability.get("heldout_probe") is True
                    and verifier_readiness.get("passed") is True
                    and verifier_readiness.get("formal_ready") is True
                    and not verifier_readiness.get("errors")
                )
                or (
                    mbo_capability.get("formal_ready") is False
                    and mbo_capability.get("independent_channel") is False
                    and mbo_capability.get("heldout_probe") is False
                    and verifier_readiness.get("passed") is False
                    and verifier_readiness.get("formal_ready") is False
                    and bool(verifier_readiness.get("errors"))
                )
            ),
            {
                "artifact": str(RADP_VERIFIER_READINESS),
                "artifact_present": RADP_VERIFIER_READINESS.is_file(),
                "passed": verifier_readiness.get("passed"),
                "formal_ready": verifier_readiness.get("formal_ready"),
                "errors": verifier_readiness.get("errors", []),
                "mbo_capability": mbo_capability,
            },
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("g0", "g1", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    protocol = load_json(PROTOCOL)
    readiness = load_json(READINESS)
    preflight = load_json(PREFLIGHT)
    checks = []
    if args.stage in {"g1", "all"}:
        checks.extend(protocol_checks(protocol))
    if args.stage in {"g0", "all"}:
        checks.extend(evidence_checks(readiness, preflight))
    passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": "sota_gate.v1",
        "stage": args.stage,
        "decision": "PASS" if passed else "STOP",
        "passed": passed,
        "api_calls": 0,
        "checks": checks,
        "claim_boundary": (
            "Zero-API readiness/protocol gate only; this artifact never authorizes a formal run "
            "without an explicit downstream approval and independent provider attestation."
        ),
        "inputs": {
            "protocol_sha256": sha256(PROTOCOL),
            "readiness_sha256": sha256(READINESS),
            "preflight_sha256": sha256(PREFLIGHT),
            "pool_sha256": sha256(POOL),
            "capabilities_sha256": sha256(CAPABILITIES),
            "freeze_sha256": sha256(FREEZE),
            "radp_verifier_readiness_sha256": sha256(RADP_VERIFIER_READINESS),
        },
    }
    output = args.output or ROOT / "experiments/results/sota_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": result["decision"], "api_calls": 0}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
