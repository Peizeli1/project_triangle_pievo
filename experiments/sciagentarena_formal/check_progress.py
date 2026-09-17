#!/usr/bin/env python3
"""Check SciAgentArena formal matrix progress and aggregate scores."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULT_ROOT = Path(__file__).resolve().parents[2] / "experiments/results/sciagentarena_drug_discovery_20260903"
TOTAL_TASKS = 79
SEEDS = (0, 1, 2)

TABLE6_METHODS = {
    ("claude_code", "base"), ("codex_cli", "base"), ("ai_researcher", "base"),
    ("researchharness", "base"), ("direct_openai", "triage_pievo"),
}
TABLE7_METHODS = {
    (h, v)
    for h in ("researchharness", "ai_researcher", "claude_code")
    for v in ("base", "self_review", "triage")
}
TABLE8_METHODS = {
    ("direct_openai", v)
    for v in (
        "triage_pievo", "triage_pievo_k1", "triage_pievo_k3",
        "triage_pievo_no_holdout", "triage_pievo_no_channel",
        "triage_pievo_no_backtrack", "triage_pievo_same_source",
    )
}


def main() -> None:
    cells_dir = RESULT_ROOT / "cells"
    if not cells_dir.is_dir():
        print("No results directory found.")
        return

    # Collect all results
    results: dict[tuple[str, str, int, str], dict] = {}
    for result_path in cells_dir.rglob("result.json"):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (data["harness"], data["variant"], data["seed"], data["task"]["task_id"])
        results[key] = data

    # Per-method aggregation
    by_method: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "complete": 0, "generation_failed": 0, "scoring_failed": 0,
        "correctness": [], "executability": [], "strategic_success": [],
    })
    for (harness, variant, seed, task_id), data in results.items():
        m = by_method[(harness, variant)]
        m[data["status"]] = m.get(data["status"], 0) + 1
        score = data.get("official_score", {}).get("score", {})
        if score:
            m["correctness"].append(score.get("correctness", 0))
            m["executability"].append(score.get("executability", 0))
            m["strategic_success"].append(score.get("strategic_success", 0))

    total_expected = len(TABLE6_METHODS | TABLE7_METHODS | TABLE8_METHODS) * TOTAL_TASKS * len(SEEDS)
    total_done = sum(1 for d in results.values() if d["status"] == "complete")

    print(f"\n{'=' * 80}")
    print(f"  SciAgentArena Formal Matrix Progress: {total_done}/{total_expected} cells")
    print(f"{'=' * 80}\n")

    # Group by table
    for table_name, table_methods in [("Table 6", TABLE6_METHODS), ("Table 7", TABLE7_METHODS), ("Table 8", TABLE8_METHODS)]:
        print(f"── {table_name} {'─' * (70 - len(table_name))}")
        print(f"  {'Method':<40s} {'Done':>6s} {'Avg Corr':>10s} {'Avg Exec':>10s}")
        for harness, variant in sorted(table_methods):
            m = by_method.get((harness, variant), {})
            done = m.get("complete", 0)
            expected = TOTAL_TASKS * len(SEEDS)
            corr = sum(m.get("correctness", [])) / max(len(m.get("correctness", [])), 1)
            exe = sum(m.get("executability", [])) / max(len(m.get("executability", [])), 1)
            pct = f"{done}/{expected}"
            print(f"  {harness}/{variant:<28s} {pct:>6s} {corr:>10.3f} {exe:>10.3f}")
        print()

    # Log tail
    log = RESULT_ROOT / "run_log.jsonl"
    if log.is_file():
        lines = log.read_text().strip().split("\n")
        print(f"── Last 5 log entries ({len(lines)} total) ──")
        for line in lines[-5:]:
            try:
                entry = json.loads(line)
                print(f"  [{entry['done']}/{entry['total']}] {entry['harness']}/{entry['variant']} seed={entry['seed']} {entry['task']} → {entry['status']}")
            except Exception:
                print(f"  {line[:120]}")
        print()


if __name__ == "__main__":
    main()
