#!/usr/bin/env python3
"""Record the MBO independent/held-out verifier readiness boundary.

This is deliberately a no-network, fail-closed probe.  It does not turn the
released surrogate into an independent evaluator; it records the exact
runtime-contract errors that must be resolved before a formal RADP run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.formal_runtime_contract import validate_runtime
except ModuleNotFoundError:  # direct script execution from the experiments dir
    from formal_runtime_contract import validate_runtime


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "official_sources/PiEvo_radp_qwen235b"
TASK_CONFIG = ROOT / "experiments/configs/official_pievo_mbo_task.yaml"
MODEL_CONFIG = ROOT / "experiments/configs/official_pievo_mbo_model_short_smiles.yaml"
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
DEFAULT_OUTPUT = ROOT / "experiments/results/radp_mbo_independent_verifier_readiness.json"


def formal_environment(seed: int) -> dict[str, str]:
    """Return the runtime settings a formal RADP MBO run must attest."""

    return {
        "PIEVO_ALGORITHM_ID": "radp_search_afe_v8",
        "PIEVO_SOURCE_GATE": "1",
        "PIEVO_SOURCE_GATE_MODE": "active",
        "PIEVO_SOURCE_GATE_PROVIDER": "official_mbo",
        "PIEVO_REVISION_PROTOCOL": "1",
        "PIEVO_ACTIVE_BACKTRACK": "1",
        "PIEVO_GUIDANCE_ONLY_CONTEXT": "1",
        "PIEVO_RADP_SEARCH_STRATEGY": "afe_v8",
        "PIEVO_MBO_INDEPENDENT_VERIFIER": "chembl37_generalized_v1",
        "PIEVO_SEED": str(seed),
    }


def assess(seed: int = 0) -> dict[str, object]:
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
    return {
        "schema_version": "radp_mbo_independent_verifier_readiness.v2",
        "task": "MBO",
        "method": "radp",
        "seed": seed,
        "passed": contract["passed"],
        "formal_ready": contract["passed"],
        "errors": contract["errors"],
        "contract": contract,
        "claim_boundary": (
            "An exact-molecule lookup or released-surrogate repeat is not a generalized "
            "verification channel for open-ended MBO. Formal MBO remains locked until "
            "two post-cutoff models, a never-trained held-out partition, calibrated "
            "applicability, representative/on-policy coverage, and complete hashed "
            "provenance all pass."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = assess(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "formal_ready": result["formal_ready"],
                "errors": result["errors"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
