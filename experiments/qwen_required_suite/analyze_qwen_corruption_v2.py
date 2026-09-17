#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/results/qwen_required_suite_v2/corruption"
OUT_JSON = BASE / "corruption_summary_v2.json"
OUT_CSV = BASE / "corruption_cells_v2.csv"
CONDITIONS = ("clean", "transient_small", "transient_large", "persistent_bias")
METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)
BUDGET = 8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    result = []
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
            result.append(row)
    return result


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def stdev(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def cell(condition: str, method: str, seed: int) -> dict[str, Any]:
    d = BASE / f"qwen235b_{method}_mbo_seed{seed}_budget{BUDGET}_{condition}"
    audit = primary_rows(d / "official_chembl_tool_audit.jsonl")
    att = {}
    ap = d / "corruption_cell_attestation.json"
    if ap.is_file():
        att = json.loads(ap.read_text(encoding="utf-8"))
    proxy_values = [float(r["output"]) for r in audit]
    raw_values = [float(r.get("raw_output", r["output"])) for r in audit]
    corrupted = [r for r in audit if r.get("corruption_applied") is True]
    proxy_best = max(proxy_values) if proxy_values else None
    raw_best = max(raw_values) if raw_values else None
    corrupted_proxy_best = bool(corrupted and proxy_best is not None and any(float(r["output"]) == proxy_best for r in corrupted))
    gate_rows = [r for r in load_jsonl(d / "radp_runtime_audit.jsonl") if r.get("event") == "gate_decision"]
    replaced = [r for r in gate_rows if r.get("decision", {}).get("action") == "replace"]
    rejected = [r for r in gate_rows if r.get("decision", {}).get("update_allowed") is False]
    corrupted_gate_decisions = []
    corrupted_inputs = {str(r.get("input")) for r in corrupted}
    for r in gate_rows:
        evidence = r.get("evidence", {})
        if str(evidence.get("candidate")) in corrupted_inputs:
            corrupted_gate_decisions.append(r)
    corrupted_replaced = sum(r.get("decision", {}).get("action") == "replace" for r in corrupted_gate_decisions)
    corrupted_allowed = sum(r.get("decision", {}).get("update_allowed") is True for r in corrupted_gate_decisions)
    effective_values = []
    if method == "radp" and gate_rows:
        for r in gate_rows:
            v = r.get("decision", {}).get("selected_outcome")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                effective_values.append(float(v))
    elif proxy_values:
        effective_values = list(proxy_values)
    effective_best = max(effective_values) if effective_values else None
    return {
        "condition": condition,
        "method": method,
        "seed": seed,
        "status": att.get("status"),
        "valid_primary": len(audit),
        "proxy_best": proxy_best,
        "raw_best": raw_best,
        "effective_best": effective_best,
        "proxy_raw_best_gap": (proxy_best - raw_best) if proxy_best is not None and raw_best is not None else None,
        "corruption_count": len(corrupted),
        "corrupted_proxy_best": corrupted_proxy_best,
        "gate_decisions": len(gate_rows),
        "gate_replace_count": len(replaced),
        "gate_reject_count": len(rejected),
        "corrupted_gate_decisions": len(corrupted_gate_decisions),
        "corrupted_replaced": corrupted_replaced,
        "corrupted_allowed": corrupted_allowed,
    }


def main() -> None:
    cells = [cell(c, m, s) for c in CONDITIONS for m in METHODS for s in SEEDS]
    clean_raw = {(r["method"], r["seed"]): r["raw_best"] for r in cells if r["condition"] == "clean"}
    for r in cells:
        base = clean_raw.get((r["method"], r["seed"]))
        if base is not None and r["raw_best"] is not None:
            r["raw_best_delta_vs_clean"] = r["raw_best"] - base
        else:
            r["raw_best_delta_vs_clean"] = None
    grouped: dict[str, Any] = {}
    for condition in CONDITIONS:
        grouped[condition] = {}
        for method in METHODS:
            subset = [r for r in cells if r["condition"] == condition and r["method"] == method]
            def vals(key: str) -> list[float]:
                return [float(r[key]) for r in subset if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
            grouped[condition][method] = {
                "n": len(subset),
                "complete_cells": sum(r["valid_primary"] == BUDGET for r in subset),
                "mean_valid_primary": mean(vals("valid_primary")),
                "proxy_best_mean": mean(vals("proxy_best")),
                "proxy_best_std": stdev(vals("proxy_best")),
                "raw_best_mean": mean(vals("raw_best")),
                "raw_best_std": stdev(vals("raw_best")),
                "effective_best_mean": mean(vals("effective_best")),
                "mean_proxy_raw_best_gap": mean(vals("proxy_raw_best_gap")),
                "mean_raw_best_delta_vs_clean": mean(vals("raw_best_delta_vs_clean")),
                "corrupted_proxy_best_rate": mean([float(r["corrupted_proxy_best"]) for r in subset if r["corruption_count"] > 0]),
                "corrupted_replaced": sum(int(r["corrupted_replaced"]) for r in subset),
                "corrupted_allowed": sum(int(r["corrupted_allowed"]) for r in subset),
                "corrupted_gate_decisions": sum(int(r["corrupted_gate_decisions"]) for r in subset),
            }
    payload = {
        "schema_version": "qwen_corruption_analysis.v2",
        "model": "qwen3-235b-a22b-instruct-2507",
        "primary_budget": BUDGET,
        "cells": cells,
        "summary": grouped,
        "interpretation_boundary": [
            "raw_best is the best uncorrupted surrogate value among actually generated candidates; it is not a physical ground truth measurement.",
            "persistent_bias is a controlled evaluator/proxy failure stress test, not natural reward hacking.",
            "RADP verification in this MBO setup repeats the same deterministic released surrogate; it detects transient inconsistency but is not an independent physical channel."
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(cells[0].keys()) if cells else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(cells)
    print(json.dumps(grouped, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
