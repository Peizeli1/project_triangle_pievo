#!/usr/bin/env python3
"""Audit the isolated Luna fixed-vs-noop source-gate ablation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BEHAVIORS = ("fixed", "noop")
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
    rows: list[dict[str, Any]] = []
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


def audit_cell(root: Path, behavior: str, seed: int) -> dict[str, Any]:
    cell = root / f"gpt56luna_radp_mbo_seed{seed}_budget6_gate_{behavior}"
    manifest = load_json(cell / "run_manifest.json", {})
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    tool_rows = load_jsonl(cell / AUDIT)
    primaries = sorted(
        [
            row for row in tool_rows
            if row.get("tool_name") == "evaluate_chembl"
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and finite(row.get("output"))
        ],
        key=lambda row: int(row.get("call_index", 0)),
    )
    gate_rows = load_jsonl(cell / GATE)
    decisions = [row for row in gate_rows if row.get("gate_terminal") is True]
    injected_indices = [2, 3]
    result: dict[str, Any] = {
        "behavior": behavior,
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
        "injection_calls": [int(row.get("call_index")) for row in primaries if row.get("corruption_applied") is True],
        "injection_offsets": [
            float(row.get("output")) - float(row.get("raw_output"))
            for row in primaries
            if row.get("corruption_applied") is True and finite(row.get("raw_output"))
        ],
        "reported_outputs": [row.get("output") for row in primaries],
        "gate_audit_rows": len(gate_rows),
        "terminal_gate_rows": len(decisions),
        "terminal_gate_behaviors": [row.get("gate_behavior") for row in decisions],
        "terminal_source_decisions": [row.get("source_decision") for row in decisions],
        "terminal_update_allowed": [row.get("update_allowed") for row in decisions],
        "terminal_raw_source_decisions": [row.get("raw_source_decision") for row in decisions],
    }
    injected_decisions = [decisions[index - 1] for index in injected_indices if len(decisions) >= index]
    if behavior == "fixed":
        behavior_checks = {
            "injected_updates_rejected": all(row.get("update_allowed") is False for row in injected_decisions),
            "injected_source_action_is_replace": all(row.get("source_decision") == "replace" for row in injected_decisions),
            "injected_raw_disagreement": all(row.get("raw_source_decision") == "verifier_channels_disagree" for row in injected_decisions),
        }
    else:
        behavior_checks = {
            "injected_updates_allowed": all(row.get("update_allowed") is True for row in injected_decisions),
            "injected_source_action_is_noop_accept": all(row.get("source_decision") == "noop_accept" for row in injected_decisions),
            "injected_raw_disagreement": all(row.get("raw_source_decision") == "verifier_channels_disagree" for row in injected_decisions),
        }
    checks = {
        "manifest_completed": manifest.get("status") == "completed",
        "identity_matches": (
            identity.get("task") == "MBO"
            and identity.get("method") == "radp"
            and identity.get("seed") == seed
            and identity.get("primary_budget") == 6
            and identity.get("model") == "gpt-5.6-luna"
        ),
        "medium_reasoning": result["runtime"]["reasoning_effort"] == "medium",
        "six_primary_evaluations": len(primaries) == 6,
        "contiguous_primary_indices": result["primary_indices"] == list(range(1, 7)),
        "same_injected_calls": result["injection_calls"] == injected_indices,
        "same_injected_offset": len(result["injection_offsets"]) == 2 and all(abs(value - 2.0) < 1e-9 for value in result["injection_offsets"]),
        "gate_enabled": result["runtime"]["source_gate_enabled"] is True,
        "revision_protocol_enabled": result["runtime"]["revision_protocol_enabled"] is True,
        "six_terminal_gate_rows": len(decisions) == 6,
        "all_gate_behavior_matches": all(value == behavior for value in result["terminal_gate_behaviors"]),
        "verification_boundary_not_overclaimed": all(
            row.get("independent_channel") is False
            and "same deterministic surrogate" in str(row.get("verification_boundary", ""))
            for row in decisions
        ),
        **behavior_checks,
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    result["claim_boundary"] = {
        "paper_eligible": False,
        "interpretation": "Controlled evaluator-side fixed-vs-noop gate ablation only.",
        "not_supported": [
            "real laboratory fault-diagnosis accuracy",
            "independent physical verification",
            "universal SOTA",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/results/exploratory_gate_ablation_gpt56luna_20260812"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cells = [audit_cell(args.root, behavior, seed) for behavior in BEHAVIORS for seed in SEEDS]
    result = {
        "schema_version": "gpt56luna_gate_behavior_ablation_audit.v1",
        "paper_eligible": False,
        "all_cells_passed": all(cell["passed"] for cell in cells),
        "cells": cells,
        "paired_mechanism_summary": {
            "fixed": "verifier disagreement rejects the injected update and repairs/quarantines internal search history",
            "noop": "the same verifier disagreement is logged but update authorization remains true",
            "injected_schedule": "calls 2 and 3, offset +2.0",
        },
        "claim_boundary": "This is a controlled mechanism ablation, not a real-world anomaly benchmark or independent physical validation.",
    }
    output = args.output or (args.root / "gate_ablation_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "all_cells_passed": result["all_cells_passed"], "cells": len(cells)}, ensure_ascii=False))
    return 0 if result["all_cells_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
