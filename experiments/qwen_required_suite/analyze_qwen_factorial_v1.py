#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/results/qwen_required_suite_v1/factorial"
REC = ROOT / "experiments/results/qwen_required_suite_v1/recovery_cap12"
INT = ROOT / "experiments/results/qwen_required_suite_v1/recovery_interruption"
OUT = ROOT / "experiments/results/qwen_required_suite_v1/factorial_analysis"
TASKS = ("NHO", "MBO", "SPO", "TMC")
ARMS = ("pievo", "triage_only", "radp_no_gate", "radp_full")
SEEDS = (0, 1, 2)
BUDGET = 8
MU_REF = {"NHO": 2.0, "MBO": 6.5, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "NHO": "official_nho_tool_audit.jsonl",
    "MBO": "official_chembl_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}
RECOVERY_CAP12 = {
    ("MBO", "pievo", 0), ("MBO", "pievo", 1), ("MBO", "pievo", 2),
    ("NHO", "triage_only", 0), ("NHO", "triage_only", 1), ("NHO", "triage_only", 2),
    ("TMC", "triage_only", 2),
}
INTERRUPTION_RECOVERY = {("NHO", "radp_full", 2)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict): rows.append(row)
    return rows


def primary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        v = row.get("output")
        if (
            str(row.get("tool_name", "")).startswith("evaluate_")
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        ):
            out.append(row)
    return sorted(out, key=lambda r: int(r.get("call_index", 0)))


def selected_dir(task: str, arm: str, seed: int) -> tuple[Path, str]:
    key = (task, arm, seed)
    name = f"qwen235b_{task.lower()}_{arm}_seed{seed}_budget{BUDGET}"
    if key in RECOVERY_CAP12:
        return REC / name, "cap12_recovery"
    if key in INTERRUPTION_RECOVERY:
        return INT / name, "interruption_recovery"
    return BASE / name, "original"


def manifest_status(d: Path) -> tuple[str | None, int | None]:
    p = d / "run_manifest.json"
    if not p.is_file(): return None, None
    try: m = json.loads(p.read_text(encoding="utf-8"))
    except ValueError: return None, None
    return m.get("status"), m.get("exit_code")


def cell(task: str, arm: str, seed: int) -> dict[str, Any]:
    d, provenance = selected_dir(task, arm, seed)
    rows = primary(load_jsonl(d / AUDIT_NAMES[task]))
    values = [float(r["output"]) for r in rows]
    running = []
    best = -math.inf
    for v in values:
        best = max(best, v); running.append(best)
    carry = list(running)
    if carry and len(carry) < BUDGET:
        carry.extend([carry[-1]] * (BUDGET - len(carry)))
    status, exit_code = manifest_status(d)
    return {
        "task": task,
        "arm": arm,
        "seed": seed,
        "provenance": provenance,
        "run_dir": str(d.resolve()),
        "manifest_status": status,
        "exit_code": exit_code,
        "valid_primary": len(values),
        "complete": len(values) == BUDGET,
        "early_terminated": 0 < len(values) < BUDGET,
        "best_value": max(values) if values else None,
        "sq_percent": max(values) / MU_REF[task] * 100.0 if values else None,
        "auoc_observed_percent": statistics.mean(running) / MU_REF[task] * 100.0 if running else None,
        "auoc_budget8_carry_percent": statistics.mean(carry) / MU_REF[task] * 100.0 if carry else None,
    }


def avg(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def sd(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [cell(t, a, s) for t in TASKS for a in ARMS for s in SEEDS]
    summary: dict[str, Any] = {}
    for arm in ARMS:
        summary[arm] = {"tasks": {}}
        task_sq_means = []
        task_auoc_means = []
        for task in TASKS:
            rows = [r for r in cells if r["arm"] == arm and r["task"] == task]
            sqs = [float(r["sq_percent"]) for r in rows if r["sq_percent"] is not None]
            auocs = [float(r["auoc_budget8_carry_percent"]) for r in rows if r["auoc_budget8_carry_percent"] is not None]
            t = {
                "n": len(rows),
                "complete": sum(bool(r["complete"]) for r in rows),
                "early_terminated": sum(bool(r["early_terminated"]) for r in rows),
                "mean_valid_primary": avg([float(r["valid_primary"]) for r in rows]),
                "sq_mean": avg(sqs), "sq_std": sd(sqs),
                "auoc_carry_mean": avg(auocs), "auoc_carry_std": sd(auocs),
            }
            summary[arm]["tasks"][task] = t
            if t["sq_mean"] is not None: task_sq_means.append(float(t["sq_mean"]))
            if t["auoc_carry_mean"] is not None: task_auoc_means.append(float(t["auoc_carry_mean"]))
        summary[arm]["macro_sq"] = avg(task_sq_means)
        summary[arm]["macro_auoc_carry"] = avg(task_auoc_means)
        summary[arm]["complete_cells"] = sum(bool(r["complete"]) for r in cells if r["arm"] == arm)
        summary[arm]["early_terminated_cells"] = sum(bool(r["early_terminated"]) for r in cells if r["arm"] == arm)
    payload = {
        "schema_version": "qwen_factorial_analysis.v1",
        "model": "qwen3-235b-a22b-instruct-2507",
        "budget": BUDGET,
        "cells": cells,
        "summary": summary,
        "analysis_policy": {
            "recovery": "use only the recovery directories predeclared in recovery_protocol.json; failed originals remain retained",
            "early_termination": "carry the last observed best forward only for AUOC-to-budget summaries; no synthetic primary outcomes are added",
            "scope": "bounded development component ablation, not a replacement for the 24-primary main table",
        },
    }
    (OUT / "factorial_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "factorial_cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
