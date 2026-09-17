#!/usr/bin/env python3
"""Audit the isolated Luna source-confusion pilot by explicit cell paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("pievo", "radp")
SEEDS = (0, 1)
AUDIT = "official_chembl_tool_audit.jsonl"
GATE = "radp_source_gate_audit.jsonl"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def primary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in rows
            if row.get("tool_name") == "evaluate_chembl"
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and finite(row.get("output"))
        ],
        key=lambda row: int(row.get("call_index", 0)),
    )


def audit_cell(root: Path, method: str, seed: int) -> dict[str, Any]:
    cell = root / f"gpt56luna_{method}_mbo_seed{seed}_budget6_source_confusion"
    manifest = load_json(cell / "run_manifest.json", {})
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    rows = load_jsonl(cell / AUDIT)
    primaries = primary(rows)
    injected = [row for row in primaries if row.get("corruption_applied") is True]
    injection_calls = [int(row.get("call_index")) for row in injected]
    injection_offsets = [float(row.get("output")) - float(row.get("raw_output")) for row in injected if finite(row.get("raw_output"))]
    result: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "cell": str(cell.resolve()),
        "manifest_status": manifest.get("status"),
        "identity": identity,
        "runtime": {
            "model": runtime.get("generation_config", {}).get("model") if isinstance(runtime.get("generation_config"), dict) else None,
            "reasoning_effort": runtime.get("generation_config", {}).get("reasoning_effort") if isinstance(runtime.get("generation_config"), dict) else None,
            "source_gate_enabled": runtime.get("source_gate_enabled"),
            "revision_protocol_enabled": runtime.get("revision_protocol_enabled"),
            "backtrack_enabled": runtime.get("backtrack_enabled"),
            "radp_search_strategy": runtime.get("radp_search_strategy"),
            "corruption_mode": runtime.get("corruption_mode"),
        },
        "primary_count": len(primaries),
        "primary_indices": [int(row.get("call_index", 0)) for row in primaries],
        "injection_calls": injection_calls,
        "injection_offsets": injection_offsets,
        "raw_outputs": [row.get("raw_output") for row in primaries],
        "reported_outputs": [row.get("output") for row in primaries],
        "duplicate_rows": sum(row.get("duplicate_suppressed") is True for row in rows),
        "audit_rows": len(rows),
    }
    if method == "radp":
        gate_rows = load_jsonl(cell / GATE)
        decisions = [
            row for row in gate_rows
            if row.get("gate_terminal") is True
        ]
        rejected = [row for row in decisions if row.get("source_decision") == "replace" and row.get("update_allowed") is False]
        result["gate"] = {
            "audit_rows": len(gate_rows),
            "terminal_rows": len(decisions),
            "rejected_update_rows": len(rejected),
            "rejected_update_calls_by_order": [
                index + 1 for index, row in enumerate(decisions)
                if row.get("source_decision") == "replace" and row.get("update_allowed") is False
            ],
            "all_terminal": len(decisions) == 6,
            "all_rejections_have_disagreement": all(
                row.get("raw_source_decision") == "verifier_channels_disagree"
                and row.get("repeat_observation") != row.get("primary_observation")
                for row in rejected
            ),
            "same_surrogate_boundary_recorded": all(
                row.get("independent_channel") is False
                and "same deterministic surrogate" in str(row.get("verification_boundary", ""))
                for row in decisions
            ),
        }
    checks = {
        "manifest_completed": manifest.get("status") == "completed",
        "identity_matches": (
            identity.get("task") == "MBO"
            and identity.get("method") == method
            and identity.get("seed") == seed
            and identity.get("primary_budget") == 6
            and identity.get("model") == "gpt-5.6-luna"
        ),
        "medium_reasoning": result["runtime"]["reasoning_effort"] == "medium",
        "six_primary_evaluations": len(primaries) == 6,
        "contiguous_primary_indices": result["primary_indices"] == list(range(1, 7)),
        "same_injected_calls": injection_calls == [2, 3],
        "same_injected_offset": len(injection_offsets) == 2 and all(abs(value - 2.0) < 1e-9 for value in injection_offsets),
    }
    if method == "radp":
        checks.update(
            {
                "gate_enabled": result["runtime"]["source_gate_enabled"] is True,
                "revision_protocol_enabled": result["runtime"]["revision_protocol_enabled"] is True,
                "gate_rejects_both_injected_updates": result["gate"]["rejected_update_rows"] >= 2,
                "gate_disagreement_evidence": result["gate"]["all_rejections_have_disagreement"],
                "verification_boundary_not_overclaimed": result["gate"]["same_surrogate_boundary_recorded"],
            }
        )
    else:
        checks["gate_disabled_for_control"] = result["runtime"]["source_gate_enabled"] is False
    result["checks"] = checks
    result["passed"] = all(checks.values())
    result["claim_boundary"] = {
        "paper_eligible": False,
        "interpretation": "Controlled evaluator-side source-confusion mechanism pilot only.",
        "not_supported": [
            "real laboratory fault-diagnosis accuracy",
            "independent physical verification",
            "universal SOTA",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/results/exploratory_source_confusion_gpt56luna_20260812"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cells = [audit_cell(args.root, method, seed) for seed in SEEDS for method in METHODS]
    result = {
        "schema_version": "gpt56luna_source_confusion_pilot_audit.v1",
        "paper_eligible": False,
        "all_cells_passed": all(cell["passed"] for cell in cells),
        "cells": cells,
        "paired_mechanism_summary": {
            "injected_schedule": "calls 2 and 3, offset +2.0",
            "control": "PiEvo receives the injected observations with gate disabled",
            "treatment": "RADP receives the same schedule and rejects both injected updates after verifier disagreement",
        },
        "claim_boundary": "This is a controlled mechanism pilot, not a real-world anomaly benchmark or independent physical validation.",
    }
    output = args.output or (args.root / "source_confusion_pilot_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "all_cells_passed": result["all_cells_passed"], "cells": len(cells)}, ensure_ascii=False))
    return 0 if result["all_cells_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
