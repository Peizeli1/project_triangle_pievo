#!/usr/bin/env python3
"""Audit all cells in the AFE/gate confound experiment without score selection."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = {
    "afe_v8_off": {"method": "radp", "strategy": "afe_v8", "gate": False, "behavior": "off"},
    "afe_v8_shadow": {"method": "radp", "strategy": "afe_v8", "gate": True, "behavior": "shadow"},
    "afe_v8_active": {"method": "radp", "strategy": "afe_v8", "gate": True, "behavior": "fixed"},
    "afe_v9_active": {"method": "radp", "strategy": "afe_v9", "gate": True, "behavior": "fixed"},
}
SEEDS = (0, 1)
BUDGET = 6


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def audit_cell(root: Path, condition: str, seed: int) -> dict[str, Any]:
    directory = root / f"gpt56luna_{condition}_mbo_seed{seed}_budget{BUDGET}"
    manifest = load_json(directory / "run_manifest.json", {})
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    tool_rows = load_jsonl(directory / "official_chembl_tool_audit.jsonl")
    primary = [
        row for row in tool_rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and finite(row.get("output"))
    ]
    primary.sort(key=lambda row: int(row.get("call_index", 0)))
    gate_rows = load_jsonl(directory / "radp_source_gate_audit.jsonl")
    afe_rows = load_jsonl(directory / "radp_afe_proposal_audit.jsonl")
    portfolio_rows = load_jsonl(directory / "radp_candidate_portfolio_audit.jsonl")
    terminal = [row for row in gate_rows if row.get("gate_terminal") is True]
    definition = CONDITIONS[condition]
    injection_offsets = [
        float(row["output"]) - float(row["raw_output"])
        for row in primary
        if row.get("corruption_applied") is True and finite(row.get("raw_output"))
    ]
    checks = {
        "manifest_completed": manifest.get("status") == "completed",
        "identity": (
            identity.get("task") == "MBO"
            and identity.get("method") == definition["method"]
            and identity.get("seed") == seed
            and identity.get("primary_budget") == BUDGET
            and identity.get("model") == "gpt-5.6-luna"
        ),
        "medium_reasoning": runtime.get("generation_config", {}).get("reasoning_effort") == "medium",
        "six_primary_evaluations": len(primary) == BUDGET,
        "contiguous_primary_indices": [int(row.get("call_index", 0)) for row in primary] == list(range(1, BUDGET + 1)),
        "same_injected_calls": [int(row.get("call_index", 0)) for row in primary if row.get("corruption_applied") is True] == [2, 3],
        "same_injected_offset": len(injection_offsets) == 2 and all(abs(value - 2.0) < 1e-9 for value in injection_offsets),
        "source_gate_matches": runtime.get("source_gate_enabled") is definition["gate"],
        "strategy_matches": runtime.get("radp_search_strategy") == definition["strategy"],
        "afe_audit_present": bool(afe_rows or portfolio_rows),
        "behavior_matches": (
            not terminal or all(row.get("gate_behavior", definition["behavior"]) == definition["behavior"] for row in terminal)
        ),
    }
    if definition["gate"]:
        checks["terminal_gate_count_if_enabled"] = len(terminal) == BUDGET
    else:
        checks["no_gate_audit_if_off"] = len(terminal) == 0
    values = [float(row["output"]) for row in primary]
    running = []
    current = -math.inf
    for value in values:
        current = max(current, value)
        running.append(current)
    metrics = {
        "sq_percent": max(values) / 6.5 * 100.0 if values else None,
        "auoc_percent": statistics.fmean(running) / 6.5 * 100.0 if running else None,
        "best_value": max(values) if values else None,
        "primary_evaluations": len(values),
        "verification_rows": len(terminal),
        "afe_audit_rows": len(afe_rows),
        "portfolio_audit_rows": len(portfolio_rows),
        "verification_cost": sum(float(row.get("verification_cost", 0.0) or 0.0) for row in terminal),
        "source_decisions": [row.get("source_decision") for row in terminal],
        "update_allowed": [row.get("update_allowed") for row in terminal],
    }
    result = {
        "condition": condition,
        "seed": seed,
        "directory": str(directory.resolve()),
        "definition": definition,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "paper_eligible": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = [audit_cell(args.root, condition, seed) for condition in CONDITIONS for seed in SEEDS]
    result = {
        "schema_version": "gpt56luna_afe_gate_modes_audit.v1",
        "paper_eligible": False,
        "cells": cells,
        "all_cells_passed": all(cell["passed"] for cell in cells),
        "selection_rule": "all eight predeclared cells audited; no score-based selection",
        "claim_boundary": "Controlled released-surrogate factor audit only; not real-world source attribution, independent chemistry verification, or universal SOTA.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "all_cells_passed": result["all_cells_passed"], "cells": len(cells)}, ensure_ascii=False))
    return 0 if result["all_cells_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
