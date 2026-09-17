#!/usr/bin/env python3
"""Summarize the paper-ineligible fixed-vs-noop gate ablation offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BEHAVIORS = ("fixed", "noop")
SEEDS = (0, 1)


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
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def summarize_cell(root: Path, behavior: str, seed: int) -> dict[str, Any]:
    cell = root / f"gpt56luna_radp_mbo_seed{seed}_budget6_gate_{behavior}"
    manifest = load_json(cell / "run_manifest.json", {})
    rows = [
        row for row in load_jsonl(cell / "official_chembl_tool_audit.jsonl")
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"verification", "duplicate_suppressed"}
        and finite(row.get("output"))
    ]
    rows.sort(key=lambda row: int(row.get("call_index", 0)))
    gate_rows = [row for row in load_jsonl(cell / "radp_source_gate_audit.jsonl") if row.get("gate_terminal") is True]
    by_call = {int(row.get("call_index", 0)): row for row in gate_rows}
    primary_values = [float(row["output"]) for row in rows]
    internal_values: list[float] = []
    for index, row in enumerate(rows, start=1):
        gate = by_call.get(index, {})
        if behavior == "fixed" and finite(gate.get("verified_observation")):
            internal_values.append(float(gate["verified_observation"]))
        else:
            internal_values.append(float(row["output"]))
    injected = [row for row in rows if row.get("corruption_applied") is True]
    rejected = [row for row in gate_rows if row.get("raw_source_decision") == "verifier_channels_disagree"]
    revision_log = load_jsonl(cell / "principle_revision_log.jsonl")
    token_usage = load_json(cell / "token_usage.json", {})
    return {
        "behavior": behavior,
        "seed": seed,
        "cell": str(cell.resolve()),
        "manifest_status": manifest.get("status"),
        "primary_count": len(rows),
        "candidate_sequence": [row.get("canonical_smiles") for row in rows],
        "reported_values": primary_values,
        "internal_search_values": internal_values,
        "reported_peak": max(primary_values) if primary_values else None,
        "internal_search_peak": max(internal_values) if internal_values else None,
        "injected_calls": [int(row.get("call_index")) for row in injected],
        "injected_reported_minus_raw": [
            float(row["output"]) - float(row["raw_output"])
            for row in injected if finite(row.get("raw_output"))
        ],
        "gate_terminal_rows": len(gate_rows),
        "verifier_disagreement_rows": len(rejected),
        "disagreement_update_allowed": [row.get("update_allowed") for row in rejected],
        "verification_cost_total": sum(
            float(row.get("verification_cost", 0) or 0)
            for row in gate_rows if finite(row.get("verification_cost", 0) or 0)
        ),
        "principle_revision_log_rows": len(revision_log),
        "token_usage": token_usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/results/exploratory_gate_ablation_gpt56luna_20260812"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cells = [summarize_cell(args.root, behavior, seed) for behavior in BEHAVIORS for seed in SEEDS]
    by_behavior: dict[str, dict[str, Any]] = {}
    for behavior in BEHAVIORS:
        selected = [cell for cell in cells if cell["behavior"] == behavior]
        peaks = [cell["internal_search_peak"] for cell in selected if cell["internal_search_peak"] is not None]
        by_behavior[behavior] = {
            "seeds": [cell["seed"] for cell in selected],
            "mean_internal_search_peak": sum(peaks) / len(peaks) if peaks else None,
            "mean_reported_peak": sum(cell["reported_peak"] for cell in selected) / len(selected) if selected else None,
            "mean_verification_cost": sum(cell["verification_cost_total"] for cell in selected) / len(selected) if selected else None,
            "all_injected_updates_allowed": all(
                all(value is True for value in cell["disagreement_update_allowed"])
                for cell in selected
            ),
            "all_injected_updates_rejected": all(
                all(value is False for value in cell["disagreement_update_allowed"])
                for cell in selected
            ),
            "candidate_sequences_identical_across_behaviors": all(
                cells[0]["candidate_sequence"] == cell["candidate_sequence"]
                for cell in cells[1:]
            ) if cells else False,
        }
    result = {
        "schema_version": "gpt56luna_gate_behavior_ablation_summary.v1",
        "paper_eligible": False,
        "include_in_main_paper_tables": False,
        "root": str(args.root.resolve()),
        "cell_count": len(cells),
        "complete_cell_count": sum(cell["manifest_status"] == "completed" for cell in cells),
        "cells": cells,
        "by_behavior": by_behavior,
        "interpretation": {
            "supported": [
                "The fixed and noop conditions receive the same injected +2 anomaly schedule.",
                "The fixed gate rejects verifier-disagreement updates while noop leaves them authorized.",
                "The fixed gate replaces injected values in internal search history; noop retains the inflated primary values.",
            ],
            "not_supported": [
                "A downstream search-quality improvement at this six-call budget, because candidate sequences were identical.",
                "Real laboratory fault-diagnosis accuracy.",
                "Independent physical verification, since the verifier is the deterministic released surrogate.",
                "Universal SOTA.",
            ],
        },
        "claim_boundary": "Controlled evaluator-side mechanism/limit experiment only; not manuscript evidence.",
    }
    output = args.output or (args.root / "gate_ablation_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "complete_cell_count": result["complete_cell_count"], "cell_count": result["cell_count"]}, ensure_ascii=False))
    return 0 if result["complete_cell_count"] == result["cell_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
