#!/usr/bin/env python3
"""Merge the original corruption arms with the v3 missing-arm runs."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/results/qwen_required_suite_v2/corruption"
NEW = ROOT / "experiments/results/qwen_required_suite_v3/corruption_factorial"
OUT_JSON = NEW / "corruption_factorial_summary_v3.json"
OUT_CSV = NEW / "corruption_factorial_cells_v3.csv"
CONDITIONS = ("clean", "transient_small", "transient_large", "persistent_bias")
ARMS = ("pievo", "triage_only", "radp_no_gate", "radp_full")
SEEDS = (0, 1, 2)
BUDGET = 8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    rows = []
    for row in load_jsonl(path):
        value = row.get("output")
        if (
            str(row.get("tool_name", "")).startswith("evaluate_")
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            rows.append(row)
    return rows


def run_dir(condition: str, arm: str, seed: int) -> Path:
    if arm in {"pievo", "radp_full"}:
        method = "pievo" if arm == "pievo" else "radp"
        return OLD / f"qwen235b_{method}_mbo_seed{seed}_budget{BUDGET}_{condition}"
    return NEW / f"qwen235b_{arm}_mbo_seed{seed}_budget{BUDGET}_{condition}"


def cell(condition: str, arm: str, seed: int) -> dict[str, Any]:
    directory = run_dir(condition, arm, seed)
    audit = primary_rows(directory / "official_chembl_tool_audit.jsonl")
    proxy_values = [float(row["output"]) for row in audit]
    raw_values = [float(row.get("raw_output", row["output"])) for row in audit]
    corrupted = [row for row in audit if row.get("corruption_applied") is True]
    gate_rows = [row for row in load_jsonl(directory / "radp_runtime_audit.jsonl") if row.get("event") == "gate_decision"]
    corrupted_inputs = {str(row.get("input")) for row in corrupted}
    corrupted_gate_rows = [
        row for row in gate_rows
        if str(row.get("evidence", {}).get("candidate")) in corrupted_inputs
    ]
    proxy_best = max(proxy_values) if proxy_values else None
    raw_best = max(raw_values) if raw_values else None
    gate_enabled = arm in {"triage_only", "radp_full"}
    if gate_enabled:
        effective_values = [
            float(row["decision"]["selected_outcome"])
            for row in gate_rows
            if isinstance(row.get("decision", {}).get("selected_outcome"), (int, float))
        ]
    else:
        effective_values = proxy_values
    effective_best = max(effective_values) if effective_values else None
    if not corrupted:
        corrupted_admitted_rate = None
    elif gate_enabled and corrupted_gate_rows:
        corrupted_admitted_rate = sum(
            row.get("decision", {}).get("update_allowed") is True
            for row in corrupted_gate_rows
        ) / len(corrupted_gate_rows)
    elif not gate_enabled:
        # Source-blind arms have no authorization event: every primary
        # observation enters their optimization history by construction.
        corrupted_admitted_rate = 1.0
    else:
        corrupted_admitted_rate = None
    return {
        "condition": condition,
        "arm": arm,
        "seed": seed,
        "run_dir": str(directory.resolve()),
        "valid_primary": len(audit),
        "proxy_best": proxy_best,
        "raw_best": raw_best,
        "effective_best": effective_best,
        "proxy_raw_best_gap": (proxy_best - raw_best) if proxy_best is not None and raw_best is not None else None,
        "corruption_count": len(corrupted),
        "corrupted_proxy_best": bool(corrupted and proxy_best is not None and any(float(row["output"]) == proxy_best for row in corrupted)),
        "corrupted_admitted_rate": corrupted_admitted_rate,
        "gate_decisions": len(gate_rows),
        "corrupted_gate_decisions": len(corrupted_gate_rows),
        "corrupted_replaced": sum(row.get("decision", {}).get("action") == "replace" for row in corrupted_gate_rows),
        "corrupted_allowed": sum(row.get("decision", {}).get("update_allowed") is True for row in corrupted_gate_rows),
    }


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def summarize(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": mean(values), "sd": sd(values)}


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]


def paired_effects(cells: list[dict[str, Any]], condition: str, metric: str) -> dict[str, Any]:
    lookup = {(row["arm"], row["seed"]): row.get(metric) for row in cells if row["condition"] == condition}
    contrasts = {
        "gate_effect_legacy": ("triage_only", "pievo"),
        "gate_effect_afe": ("radp_full", "radp_no_gate"),
        "afe_effect_gate_off": ("radp_no_gate", "pievo"),
        "afe_effect_gate_on": ("radp_full", "triage_only"),
    }
    result: dict[str, Any] = {}
    for name, (left, right) in contrasts.items():
        values = [
            float(lookup[(left, seed)]) - float(lookup[(right, seed)])
            for seed in SEEDS
            if isinstance(lookup.get((left, seed)), (int, float)) and isinstance(lookup.get((right, seed)), (int, float))
        ]
        result[name] = summarize(values)
    interactions = []
    for seed in SEEDS:
        keys = [(arm, seed) for arm in ARMS]
        if all(isinstance(lookup.get(key), (int, float)) for key in keys):
            interactions.append(
                float(lookup[("radp_full", seed)])
                - float(lookup[("radp_no_gate", seed)])
                - float(lookup[("triage_only", seed)])
                + float(lookup[("pievo", seed)])
            )
    result["afe_gate_interaction"] = summarize(interactions)
    return result


def main() -> None:
    cells = [cell(condition, arm, seed) for condition in CONDITIONS for arm in ARMS for seed in SEEDS]
    clean_raw = {(row["arm"], row["seed"]): row["raw_best"] for row in cells if row["condition"] == "clean"}
    for row in cells:
        base = clean_raw.get((row["arm"], row["seed"]))
        row["raw_best_delta_vs_clean"] = (
            float(row["raw_best"]) - float(base)
            if isinstance(row.get("raw_best"), (int, float)) and isinstance(base, (int, float))
            else None
        )
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for arm in ARMS:
            subset = [row for row in cells if row["condition"] == condition and row["arm"] == arm]
            corrupted_subset = [row for row in subset if row["corruption_count"] > 0]
            summary[condition][arm] = {
                "cells": len(subset),
                "complete_cells": sum(row["valid_primary"] == BUDGET for row in subset),
                "valid_primary": summarize(numeric(subset, "valid_primary")),
                "proxy_best": summarize(numeric(subset, "proxy_best")),
                "raw_best": summarize(numeric(subset, "raw_best")),
                "effective_best": summarize(numeric(subset, "effective_best")),
                "proxy_raw_best_gap": summarize(numeric(subset, "proxy_raw_best_gap")),
                "raw_best_delta_vs_clean": summarize(numeric(subset, "raw_best_delta_vs_clean")),
                "corrupted_proxy_best_rate": summarize([float(row["corrupted_proxy_best"]) for row in corrupted_subset]),
                "corrupted_admitted_rate": summarize(numeric(corrupted_subset, "corrupted_admitted_rate")),
                "corrupted_replaced": sum(int(row["corrupted_replaced"]) for row in subset),
                "corrupted_gate_decisions": sum(int(row["corrupted_gate_decisions"]) for row in subset),
            }
        summary[condition]["factorial_effects"] = {
            metric: paired_effects(cells, condition, metric)
            for metric in ("proxy_best", "raw_best", "effective_best", "proxy_raw_best_gap", "raw_best_delta_vs_clean")
        }
    payload = {
        "schema_version": "qwen_corruption_factorial_analysis.v3",
        "model": "qwen3-235b-a22b-instruct-2507",
        "primary_budget": BUDGET,
        "arms": list(ARMS),
        "conditions": list(CONDITIONS),
        "cells": cells,
        "summary": summary,
        "claim_boundary": "controlled released-surrogate corruption factorial; paired effects use seed as the statistical unit",
    }
    NEW.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0].keys()))
        writer.writeheader()
        writer.writerows(cells)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
