#!/usr/bin/env python3
"""Audit invalid attempts and frozen structural-envelope violations in MBO runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from experiments.mbo_applicability_domain import assess_mbo_applicability


METHODS = ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2", "ai_researcher", "piflow", "pievo", "radp")
TASK = "MBO"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def primary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and isinstance(row.get("output"), (int, float))
        and not isinstance(row.get("output"), bool)
    ]


def cell(root: Path, method: str, seed: int, track: str) -> dict[str, Any]:
    run_dir = root / f"gpt56luna_nothinking_{method}_mbo_seed{seed}_formal24"
    rows = load_jsonl(run_dir / "official_chembl_tool_audit.jsonl")
    selected = primary(rows)
    ood: list[dict[str, Any]] = []
    for row in selected:
        candidate = row.get("canonical_smiles") or row.get("input")
        result = assess_mbo_applicability(str(candidate))
        if result.get("accepted") is not True:
            ood.append(
                {
                    "call_index": row.get("call_index"),
                    "output": row.get("output"),
                    "candidate": candidate,
                    "violations": result.get("violations", []),
                    "features": result.get("features", {}),
                }
            )
    ood_indices = {item.get("call_index") for item in ood}
    in_domain = [float(row["output"]) for row in selected if row.get("call_index") not in ood_indices]
    values = [float(row["output"]) for row in selected]
    return {
        "track": track,
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "primary": len(selected),
        "invalid_attempts": sum(row.get("success") is not True for row in rows if str(row.get("tool_name", "")).startswith("evaluate_")),
        "duplicates": sum(row.get("duplicate_suppressed") is True for row in rows),
        "best_output": max(values) if values else None,
        "in_domain_best_output": max(in_domain) if in_domain else None,
        "ood_primary": len(ood),
        "ood_best_output": max((float(item["output"]) for item in ood), default=None),
        "ood_examples": sorted(ood, key=lambda item: float(item.get("output") or -1), reverse=True)[:3],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    cells: list[dict[str, Any]] = []
    for method in METHODS:
        root = args.formal_root if method in {"pievo", "radp"} else args.baseline_root
        track = "pievo_radp_formal" if method in {"pievo", "radp"} else "baseline_adaptation"
        for seed in range(3):
            cells.append(cell(root, method, seed, track))

    by_method: dict[str, Any] = {}
    for method in METHODS:
        selected = [item for item in cells if item["method"] == method]
        ood_counts = [item["ood_primary"] for item in selected]
        best_values = [item["best_output"] for item in selected if item["best_output"] is not None]
        in_domain_values = [item["in_domain_best_output"] for item in selected if item["in_domain_best_output"] is not None]
        by_method[method] = {
            "cells": len(selected),
            "ood_primary_total": sum(ood_counts),
            "ood_primary_rate": sum(ood_counts) / (24 * len(selected)),
            "invalid_attempt_total": sum(item["invalid_attempts"] for item in selected),
            "duplicate_total": sum(item["duplicates"] for item in selected),
            "best_output_mean": statistics.fmean(best_values) if best_values else None,
            "in_domain_best_output_mean": statistics.fmean(in_domain_values) if in_domain_values else None,
        }

    output = {
        "schema_version": "gpt56luna_nothinking_mbo_quality_audit.v1",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "enable_thinking": False,
        "task": TASK,
        "seeds": [0, 1, 2],
        "primary_budget_per_cell": 24,
        "applicability_domain_used_during_runs": "none",
        "audit_contract": "mbo_structural_envelope_v2",
        "cells": cells,
        "by_method": by_method,
        "claim_boundary": "Post-hoc structural-envelope audit of no-thinking matched-evaluator runs; it does not retroactively convert an unfiltered run into an AD-constrained benchmark.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# GPT-5.6-Luna No-Thinking MBO Quality Audit",
        "",
        "The runs used `applicability_domain=none`; the frozen structural envelope is applied here only as a post-hoc diagnostic.",
        "",
        "| Method | OOD primary / 72 | OOD rate | Invalid attempts | Duplicates | Mean best | Mean in-domain best |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"vanilla_mas": "Vanilla MAS", "react": "ReAct", "ai_scientist_v1": "AI Scientist v1", "ai_scientist_v2": "AI Scientist v2", "ai_researcher": "AI Researcher", "piflow": "PiFlow", "pievo": "PiEvo", "radp": "RADP"}
    for method in METHODS:
        item = by_method[method]
        lines.append(
            f"| {labels[method]} | {item['ood_primary_total']} | {item['ood_primary_rate']:.3f} | {item['invalid_attempt_total']} | {item['duplicate_total']} | {item['best_output_mean']:.3f} | {item['in_domain_best_output_mean']:.3f} |"
        )
    lines.extend([
        "",
        "The audit rejects using OOD-driven MBO scores as a clean SOTA claim. Invalid attempts were rejected by the evaluator and did not consume primary budget.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json.resolve()), "output_md": str(args.output_md.resolve()), "by_method": by_method}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
