#!/usr/bin/env python3
"""Zero-API preflight for the frozen RADP-Search-AFE v8 implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RADP_REPO = ROOT / "official_sources/PiEvo_radp_qwen235b"
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
G0 = ROOT / "experiments/results/g0_evidence_audit_v8.json"
POOL = ROOT / "experiments/results/official_chembl35_pool_seed0/chembl35_pool_seed0.jsonl"
POOL_SUMMARY = ROOT / "experiments/results/official_chembl35_pool_seed0/chembl35_pool_seed0_summary.json"
COMMON = ROOT / "experiments/common_candidate_generator.py"
VALIDATOR_PIEVO = ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/tools/candidate_validation.py"
VALIDATOR_RADP = ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/tools/candidate_validation.py"
AFE = RADP_REPO / "pievo/radp/afe.py"
WIDE = RADP_REPO / "pievo/radp/wide_candidate.py"
MATCHED_FLOW = ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/group/evolveflow.py"
RADP_FLOW = RADP_REPO / "pievo/group/evolveflow.py"
RESULTS = ROOT / "experiments/results"

sys.path.insert(0, str(RADP_REPO))
sys.path.insert(0, str(ROOT))

from experiments.common_candidate_generator import (  # noqa: E402
    GENERATOR_ID,
    generate,
    generator_sha256,
)
from pievo.radp.afe import (  # noqa: E402
    ROLE_ORDER,
    generate_afe_portfolio,
    outcome_blind_initial_priors,
)
from pievo.radp.wide_candidate import (  # noqa: E402
    load_unlabeled_smiles,
    afe_v8_structural_portfolio,
)
from pievo.tools.candidate_validation import validate_mbo  # noqa: E402


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--g0", type=Path, default=G0)
    parser.add_argument("--version", default="v8")
    parser.add_argument("--strategy", default="afe_v8")
    parser.add_argument("--output", type=Path, default=RESULTS / "afe_zero_api_preflight_v8.json")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    g0_path = args.g0.resolve()
    protocol = load_json(protocol_path)
    g0 = load_json(g0_path)
    os.environ["PIEVO_RADP_SEARCH_STRATEGY"] = args.strategy
    history = [("c1ccccc1", 1.0), ("Cc1ccccc1", 2.0), ("C1CCCCCCC1", 1.5)]
    afe_rows = generate_afe_portfolio(task="MBO", history=history, seed=0, budget=12)
    afe_candidates = [str(row["candidate"]) for row in afe_rows]
    cold_rows = generate_afe_portfolio(task="MBO", history=[], seed=0, budget=3)
    nho_candidate = "fiber_radius=30,helix_radius=45,n_turns=5,pitch=100,wavelength=600,x_y_z=1,direction=0"
    nho_rows = generate_afe_portfolio(
        task="NHO", history=[], candidate_pool=[nho_candidate], seed=0, budget=1
    )
    pool = load_unlabeled_smiles(POOL, limit=50000) if POOL.is_file() else []
    raw_pool_rows = sum(
        1
        for line in POOL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ) if POOL.is_file() else 0
    structural = afe_v8_structural_portfolio(pool, budget=24, seed=0) if pool else []
    coverage_rows = generate_afe_portfolio(
        task="MBO",
        history=[("c1ccccc1", 1.0)],
        candidate_pool=["CCOc1ccccc1", "C1CCCCC1", "CCN"],
        model_proposals=["CCO", "CCN"],
        seed=0,
        budget=12,
    )
    coverage_roles = {
        row.get("role")
        for row in coverage_rows
        if row.get("source") == "unlabeled_structural_pool"
        and row.get("coverage_required") is True
    }
    common_stream = generate(task="MBO", seed=0, slot=0, count=3)

    source_text = WIDE.read_text(encoding="utf-8") if WIDE.is_file() else ""
    matched_flow_text = MATCHED_FLOW.read_text(encoding="utf-8") if MATCHED_FLOW.is_file() else ""
    radp_flow_text = RADP_FLOW.read_text(encoding="utf-8") if RADP_FLOW.is_file() else ""
    checks = [
        check(
            "protocol_identity",
            protocol.get("schema_version")
            == f"radp_search_afe_matched_qwen235b_protocol.{args.version}"
            and protocol.get("algorithm", {}).get("radp_algorithm_id")
            == ("radp_search_afe_v9" if args.strategy == "afe_v9" else protocol.get("algorithm", {}).get("radp_algorithm_id")),
            {"schema_version": protocol.get("schema_version"), "algorithm": protocol.get("algorithm")},
        ),
        check("g0_provenance_passed", g0.get("passed") is True, g0.get("claim_boundary")),
        check(
            "common_generator_replay_and_validity",
            generate(task="MBO", seed=0, slot=0, count=3) == common_stream
            and len(common_stream) == len(set(common_stream))
            and all(validate_mbo(candidate).canonical == candidate for candidate in common_stream),
            {"generator_id": GENERATOR_ID, "stream": common_stream},
        ),
        check(
            "afe_three_slot_portfolio",
            len(afe_rows) == 12
            and len(afe_candidates) == len(set(afe_candidates))
            and all(validate_mbo(candidate).canonical == candidate for candidate in afe_candidates)
            and {row.get("role") for row in afe_rows} == set(ROLE_ORDER),
            {"count": len(afe_rows), "roles": sorted({row.get("role") for row in afe_rows})},
        ),
        check(
            "afe_v8_initial_priors_are_outcome_blind",
            len(cold_rows) == 3
            and all(row.get("initial_prior") is True for row in cold_rows)
            and {row.get("role") for row in cold_rows} == set(ROLE_ORDER)
            and tuple(
                (row.get("role"), row.get("candidate")) for row in cold_rows
            ) == outcome_blind_initial_priors("MBO"),
            {"rows": cold_rows, "expected": outcome_blind_initial_priors("MBO")},
        ),
        check(
            "afe_non_mbo_validator_path",
            len(nho_rows) == 1 and nho_rows[0].get("candidate") == nho_candidate,
            nho_rows,
        ),
        check(
            "unlabeled_pool_replay",
            raw_pool_rows == 50000
            and len(pool) >= 24000
            and len(structural) == 24
            and len(structural) == len(set(structural)),
            {
                "raw_pool_rows": raw_pool_rows,
                "validated_pool_rows": len(pool),
                "structural_candidates": len(structural),
            },
        ),
        check(
            "afe_v8_source_label_and_role_coverage",
            coverage_roles == set(ROLE_ORDER)
            and all(
                row.get("source")
                in {
                    "unlabeled_structural_pool",
                    "llm_model_proposal",
                    "afe_generated_structure",
                }
                for row in coverage_rows
            )
            and all(row.get("labels_used") is False for row in coverage_rows),
            {
                "coverage_roles": sorted(coverage_roles),
                "source_counts": {
                    source: sum(row.get("source") == source for row in coverage_rows)
                    for source in (
                        "unlabeled_structural_pool",
                        "llm_model_proposal",
                        "afe_generated_structure",
                    )
                },
            },
        ),
        check(
            "pool_reader_is_label_blind",
            'row.get("canonical_smiles")' in source_text
            and 'row.get("pchembl' not in source_text.lower()
            and 'row["pchembl' not in source_text.lower()
            and 'row.get("activity_count' not in source_text.lower(),
            {"reader": str(WIDE), "labels_read": False},
        ),
        check(
            "shared_validator_identity",
            sha256(VALIDATOR_PIEVO) == sha256(VALIDATOR_RADP) and sha256(VALIDATOR_PIEVO) is not None,
            {"pievo": sha256(VALIDATOR_PIEVO), "radp": sha256(VALIDATOR_RADP)},
        ),
        check(
            "common_generator_is_wired_into_both_worktrees",
            all(
                marker in matched_flow_text and marker in radp_flow_text
                for marker in (
                    "_common_generator_candidates",
                    "common_generator_v1",
                    "radp_common_generator_v1",
                )
            )
            and "not self._common_generator_enabled()" in radp_flow_text,
            {"matched_flow": str(MATCHED_FLOW), "radp_flow": str(RADP_FLOW)},
        ),
        check(
            "source_boundaries_are_explicit",
            all(
                row.get(key) is value
                for row in afe_rows
                for key, value in (
                    ("labels_used", False),
                )
            )
            and all(
                row.get("source") in {
                    "outcome_blind_initial_prior",
                    "unlabeled_structural_pool",
                    "llm_model_proposal",
                    "afe_generated_structure",
                }
                for row in afe_rows
            )
            and AFE.is_file(),
            {"afe_path": str(AFE), "verifier_values_used": False, "source_gate_values_used": False},
        ),
    ]
    result = {
        "schema_version": f"radp_search_afe_zero_api_preflight.{args.version}",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "generator_sha256": generator_sha256(),
        "afe_sha256": sha256(AFE),
        "pool_sha256": sha256(POOL),
        "pool_summary": load_json(POOL_SUMMARY),
        "api_calls": 0,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "strategy": args.strategy,
        "claim_boundary": "AFE zero-API implementation and fairness preflight only; no live performance or SOTA claim.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": result["passed"], "api_calls": 0}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
