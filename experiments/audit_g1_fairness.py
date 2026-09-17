#!/usr/bin/env python3
"""Zero-API G1 fairness and budget audit for the matched comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from experiments.common_candidate_generator import GENERATOR_ID, generate, generator_sha256
except ModuleNotFoundError:
    from common_candidate_generator import GENERATOR_ID, generate, generator_sha256


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
G0 = RESULTS / "g0_evidence_audit_v8.json"
GENERATOR = ROOT / "experiments/common_candidate_generator.py"
VALIDATOR_RADP = ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/tools/candidate_validation.py"
VALIDATOR_PIEVO = ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/tools/candidate_validation.py"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--g0", type=Path, default=G0)
    parser.add_argument("--version", default="v8")
    parser.add_argument("--output", type=Path, default=RESULTS / "g1_fairness_audit_v8.json")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    g0_path = args.g0.resolve()
    protocol = load(protocol_path)
    g0 = load(g0_path)
    fairness = protocol.get("g1_fairness_protocol", {})
    arms = fairness.get("formal_comparison", fairness.get("formal_primary_comparison", {})).get("arms", [])
    common_arms = [
        arm
        for arm in fairness.get("zero_api_comparison_arms", [])
        if isinstance(arm, dict) and arm.get("arm_id") in {"pievo_common_generator", "radp_common_generator"}
    ]
    shared = fairness.get("shared_controls", {})
    budget = fairness.get("budget_accounting", {})
    common = fairness.get("common_generator", {})
    replay_probe = []
    try:
        replay_probe = [
            generate(task="MBO", seed=seed, slot=slot, count=3)
            for seed, slot in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
        ]
    except Exception as exc:
        replay_error = f"{type(exc).__name__}: {exc}"
    else:
        replay_error = None
    generator_subprocess = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            (
                "from experiments.common_candidate_generator import replay; "
                "assert replay(task='MBO', seed=0)==replay(task='MBO', seed=0)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    arm_ids = [arm.get("arm_id") if isinstance(arm, dict) else arm for arm in arms]
    common_generator_ids = [
        arm.get("generator_id") for arm in common_arms if isinstance(arm, dict)
    ]
    validator_hashes = {"pievo": sha256(VALIDATOR_PIEVO), "radp": sha256(VALIDATOR_RADP)}
    task_model_paths = sorted(
        list((ROOT / "experiments/configs").glob("official_pievo_*_task.yaml"))
        + list((ROOT / "experiments/configs").glob("official_pievo_*_model*.yaml"))
    )
    checks = {
        "g0_passed": g0.get("passed") is True,
        "protocol_schema_matches_version": protocol.get("schema_version")
        == f"radp_search_afe_matched_qwen235b_protocol.{args.version}",
        "g1_schema": fairness.get("schema_version") == "radp_search_afe_g1_fairness.v1",
        "strongest_pievo_reference_present": fairness.get("strongest_pievo_reference", {}).get("formal_role") == "reference_only",
        "two_formal_common_generator_arms": arm_ids == ["strongest_pievo", "radp_afe"],
        "same_common_generator_id": common_generator_ids == [GENERATOR_ID, GENERATOR_ID]
        and common.get("generator_id") == GENERATOR_ID,
        "common_generator_exists_and_hashable": GENERATOR.is_file() and generator_sha256() == sha256(GENERATOR),
        "common_generator_replay_subprocess": generator_subprocess.returncode == 0 and replay_error is None,
        "common_generator_outcome_blind": all(
            common.get(name) is True
            for name in (
                "outcome_blind",
                "uses_primary_labels",
                "uses_verifier_values",
                "uses_source_gate_decisions",
                "uses_experimenter_reasoning",
            )
        ) if False else (
            common.get("outcome_blind") is True
            and common.get("uses_primary_labels") is False
            and common.get("uses_verifier_values") is False
            and common.get("uses_source_gate_decisions") is False
            and common.get("uses_experimenter_reasoning") is False
        ),
        "same_candidate_validator_for_common_arms": shared.get("same_candidate_validator") is True
        and validator_hashes["pievo"] == validator_hashes["radp"]
        and validator_hashes["pievo"] is not None,
        "same_candidate_space": shared.get("candidate_space_same_for_common_arms") is True,
        "same_model_prompt_task_and_primary_budget": shared.get("model") == protocol.get("model")
        and protocol.get("primary_budget") == 24
        and common.get("same_ordered_stream_for_common_arms") is True,
        "seed_order_is_counterbalanced": shared.get("counterbalanced_method_order") == {
            "seed0": ["pievo_common_generator", "radp_common_generator"],
            "seed1": ["radp_common_generator", "pievo_common_generator"],
            "seed2": ["pievo_common_generator", "radp_common_generator"],
        },
        "budget_accounting_is_explicit": budget.get("primary_budget_equal") is True
        and budget.get("verification_cost_recorded_separately") is True
        and budget.get("equal_total_action_budget_required_for_cost_sweep") is True,
        "formal_matrix_is_144_mbo_primary": budget.get("mbo_formal_matrix")
        == "2 methods x 3 seeds x 24 primary = 144 primary",
        "task_and_model_configs_present": len(task_model_paths) == 8
        and all(path.is_file() for path in task_model_paths),
        "no_partial_or_retry_splicing": shared.get("no_retry_or_partial_splicing") is True,
        "no_label_or_observed_score_leakage": shared.get("no_test_label_or_observed_score_in_generator_context") is True,
    }
    result = {
        "schema_version": f"g1_fairness_audit.{args.version}",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "g0_path": str(g0_path),
        "g0_sha256": sha256(g0_path),
        "generator_sha256": sha256(GENERATOR),
        "validator_sha256": validator_hashes,
        "replay_probe": replay_probe,
        "replay_error": replay_error,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": "G1 fairness/budget protocol only; no API authorization and no performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": result["passed"], "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
