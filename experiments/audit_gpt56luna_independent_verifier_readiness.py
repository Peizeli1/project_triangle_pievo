#!/usr/bin/env python3
"""Fail-closed readiness audit for an independent Luna chemistry verifier.

This audit only checks local contracts.  It never turns the released ChEMBL
surrogate into independent evidence and never launches an API run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from experiments.formal_runtime_contract import validate_runtime
    from experiments.independent_verifier_contract import load_and_validate_provider_attestation
except ModuleNotFoundError:  # direct execution from the experiments directory
    from formal_runtime_contract import validate_runtime
    from independent_verifier_contract import load_and_validate_provider_attestation


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "official_sources/PiEvo_radp_qwen235b"
TASK_CONFIG = ROOT / "experiments/configs/official_pievo_mbo_task.yaml"
MODEL_CONFIG = ROOT / "experiments/configs/official_pievo_mbo_model_short_smiles.yaml"
PROTOCOL = ROOT / "experiments/configs/gpt56luna_medium_pievo_radp_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments/results/gpt56luna_independent_verifier_readiness_strict_20260813.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def formal_environment(seed: int) -> dict[str, str]:
    return {
        "PIEVO_ALGORITHM_ID": "radp_search_afe_v9",
        "PIEVO_SOURCE_GATE": "1",
        "PIEVO_SOURCE_GATE_MODE": "active",
        "PIEVO_SOURCE_GATE_PROVIDER": "official_mbo",
        "PIEVO_REVISION_PROTOCOL": "1",
        "PIEVO_ACTIVE_BACKTRACK": "1",
        "PIEVO_GUIDANCE_ONLY_CONTEXT": "1",
        "PIEVO_RADP_SEARCH_STRATEGY": "afe_v9",
        "PIEVO_MBO_INDEPENDENT_VERIFIER": "chembl37_generalized_v1",
        "PIEVO_PREFILTER_INVALID_CANDIDATES": "1",
        "PIEVO_MAX_PREFILTER_REJECTIONS": "12",
        "PIEVO_MAX_PREFILTER_EMPTY_POOLS": "2",
        "PIEVO_SEED": str(seed),
    }


def assess(seed: int) -> dict[str, Any]:
    contract = validate_runtime(
        method="radp",
        task="MBO",
        seed=seed,
        repository=REPOSITORY,
        environment=formal_environment(seed),
        input_paths={
            "task_config": TASK_CONFIG,
            "model_config": MODEL_CONFIG,
            "protocol": PROTOCOL,
        },
        require_formal_ready=True,
    )
    errors = list(contract.get("errors", []))
    capabilities = load_json(REPOSITORY / "pievo/radp/capabilities.json")
    task_capabilities = capabilities.get("tasks", {}).get("MBO", {})
    if task_capabilities.get("independent_channel") is not True:
        errors.append("MBO capability manifest does not provide independent_channel=true")
    if task_capabilities.get("heldout_probe") is not True:
        errors.append("MBO capability manifest does not provide heldout_probe=true")
    attestation_path_value = task_capabilities.get("provider_attestation_path")
    provider_check: dict[str, Any]
    if not isinstance(attestation_path_value, str) or not attestation_path_value.strip():
        provider_check = {
            "passed": False,
            "errors": ["provider attestation path is missing"],
        }
        errors.append("no independent verifier provider attestation is registered")
    else:
        attestation_path = Path(attestation_path_value)
        if not attestation_path.is_absolute():
            attestation_path = REPOSITORY / attestation_path
        provider_check = load_and_validate_provider_attestation(
            attestation_path,
            task="MBO",
            provider="chembl37_generalized_v1",
        )
        if provider_check.get("passed") is not True:
            errors.append("registered independent verifier attestation failed closed")

    return {
        "schema_version": "gpt56luna_independent_verifier_readiness.v1",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "task": "MBO",
        "method": "radp",
        "seed": seed,
        "passed": not errors,
        "formal_ready": not errors,
        "errors": errors,
        "runtime_contract": contract,
        "provider_attestation": provider_check,
        "claim_boundary": {
            "supported": "The local protocol and independent-verifier requirements are audited fail-closed.",
            "not_supported": [
                "Independent physical chemistry verification",
                "A generalized verifier claim from an exact-molecule lookup or released surrogate",
                "Formal RADP runs that use a verifier not covered by the attestation contract",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = assess(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "passed": result["passed"],
        "formal_ready": result["formal_ready"],
        "errors": result["errors"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
