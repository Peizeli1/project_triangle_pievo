#!/usr/bin/env python3
"""Merge legacy and AFE proxy arms and report seed-level uncertainty."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/results/qwen_required_suite_v2/proxy_exploitation/results.json"
NEW = ROOT / "experiments/results/qwen_required_suite_v3/proxy_factorial"
AFE = NEW / "afe_runs.json"
OUT_JSON = NEW / "proxy_factorial_summary_v2.json"
OUT_CSV = NEW / "proxy_factorial_seed_metrics_v2.csv"
ARMS = ("legacy_no_gate", "legacy_gate", "afe_no_gate", "afe_gate")
METRICS = (
    "proxy_best", "true_best", "model_round_true_mean", "model_round_proxy_mean",
    "exploit_rate", "false_admissions", "quarantined_exploits",
    "verification_cost", "cumulative_true_regret",
)


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
    }


def paired_effects(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    lookup = {(row["arm"], int(row["seed"])): float(row[metric]) for row in rows}
    seeds = sorted({seed for _, seed in lookup})
    contrasts = {
        "gate_effect_legacy": ("legacy_gate", "legacy_no_gate"),
        "gate_effect_afe": ("afe_gate", "afe_no_gate"),
        "afe_effect_gate_off": ("afe_no_gate", "legacy_no_gate"),
        "afe_effect_gate_on": ("afe_gate", "legacy_gate"),
    }
    result = {}
    for name, (left, right) in contrasts.items():
        values = [lookup[(left, seed)] - lookup[(right, seed)] for seed in seeds if (left, seed) in lookup and (right, seed) in lookup]
        result[name] = summarize(values)
    interactions = [
        lookup[("afe_gate", seed)] - lookup[("afe_no_gate", seed)]
        - lookup[("legacy_gate", seed)] + lookup[("legacy_no_gate", seed)]
        for seed in seeds
        if all((arm, seed) in lookup for arm in ARMS)
    ]
    result["afe_gate_interaction"] = summarize(interactions)
    return result


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(AFE.read_text(encoding="utf-8"))
    mapping = {"source_blind": "legacy_no_gate", "triage": "legacy_gate"}
    runs = []
    for run in old["runs"]:
        copied = dict(run)
        copied["arm"] = mapping[run["arm"]]
        runs.append(copied)
    runs.extend(new["runs"])
    runs.sort(key=lambda row: (row["arm"], int(row["seed"])))
    seed_rows = [
        {"arm": run["arm"], "seed": int(run["seed"]), **{metric: run["metrics"][metric] for metric in METRICS}}
        for run in runs
    ]
    summary = {
        arm: {
            metric: summarize([float(row[metric]) for row in seed_rows if row["arm"] == arm])
            for metric in METRICS
        }
        for arm in ARMS
    }
    effects = {metric: paired_effects(seed_rows, metric) for metric in METRICS}
    payload = {
        "schema_version": "qwen_proxy_factorial_analysis.v2",
        "model": old["model"],
        "seeds": old["seeds"],
        "rounds": old["rounds"],
        "arms": list(ARMS),
        "runs": runs,
        "seed_metrics": seed_rows,
        "summary": summary,
        "factorial_effects": effects,
        "claim_boundary": "controlled one-dimensional proxy mismatch; uncertainty is computed across seeds, not by treating within-seed decisions as independent",
    }
    NEW.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)
    print(json.dumps({"summary": summary, "factorial_effects": effects}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
