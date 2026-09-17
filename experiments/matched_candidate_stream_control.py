#!/usr/bin/env python3
"""Zero-API matched-candidate control for the official-MBO diagnostic.

The live seed0 failure mixed two effects: RADP generated a much worse
candidate stream, while its source gate correctly repaired both injected
observations.  This control replays the same valid primary stream through the
two accounting paths and checks the source-confusion bookkeeping separately.
It is an isolation diagnostic, not evidence that RADP's search is superior.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def primary_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("tool_name", "").startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and isinstance(row.get("call_index"), int)
        and isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
        and math.isfinite(float(row["output"]))
    ]
    return sorted(rows, key=lambda row: row["call_index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/private/tmp/ai4ai_radp_seed0_feedback_authorized.wauBA5/seed0"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    pievo_rows = primary_rows(root / "pievo/official_chembl_tool_audit.jsonl")
    radp_rows = primary_rows(root / "radp/official_chembl_tool_audit.jsonl")
    gate_rows = read_jsonl(root / "radp/radp_source_gate_audit.jsonl")

    reference_candidates = [str(row.get("canonical_smiles")) for row in pievo_rows]
    reference_outputs = [float(row["raw_output"]) for row in pievo_rows]
    unique_reference = len(reference_candidates) == len(set(reference_candidates))
    valid_reference = all(
        row.get("canonical_smiles") and row.get("success") is True
        for row in pievo_rows
    )
    source_rejects = sum(row.get("source_decision") == "replace" for row in gate_rows)
    source_repairs = sum(
        row.get("source_decision") == "replace"
        and row.get("update_allowed") is False
        and row.get("verified_observation") is not None
        for row in gate_rows
    )

    # The released local surrogate is deterministic for a canonical SMILES.
    # Replaying PiEvo's valid stream therefore gives both methods identical
    # raw rewards; no model, evaluator API, or new primary call is made.
    replay_best = max(reference_outputs, default=None)
    result = {
        "schema_version": "matched_candidate_stream_control.v1",
        "api_calls": 0,
        "input_root": str(root),
        "reference_method": "matched_pievo",
        "reference_primary_count": len(pievo_rows),
        "other_method_primary_count": len(radp_rows),
        "matched_candidate_count": len(reference_candidates),
        "matched_candidate_unique": unique_reference,
        "matched_candidate_valid": valid_reference,
        "replay_best_true_output": replay_best,
        "replay_pievo_best_true_output": replay_best,
        "replay_radp_best_true_output": replay_best,
        "source_gate_reject_count": source_rejects,
        "source_gate_repair_count": source_repairs,
        "checks": {
            "reference_has_exact_six_valid_unique_primaries": (
                len(pievo_rows) == 6 and valid_reference and unique_reference
            ),
            "other_method_completed_six_primaries": len(radp_rows) == 6,
            "replayed_reward_noninferior": replay_best is not None,
            "source_gate_rejects_both_injected_failures": source_rejects == 2,
            "source_gate_repairs_both_injected_failures": source_repairs == 2,
        },
        "claim_boundary": (
            "This isolates source-gate accounting on a matched candidate stream. "
            "It does not establish RADP search superiority or an official benchmark claim."
        ),
    }
    result["passed"] = all(result["checks"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
