#!/usr/bin/env python3
"""Summarize the controlled ResearchHarness Base/Self-Review/Triage run."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ARMS = ("vanilla", "self_review", "gate")
NAMES = {"vanilla": "Base", "self_review": "+ Self-Review", "gate": "+ Triage"}


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def trace_metrics(trace_dir: Path) -> dict[str, float]:
    traces = list(trace_dir.rglob("trace_*.jsonl"))
    if len(traces) != 1:
        raise RuntimeError(f"expected one trace in {trace_dir}, found {len(traces)}")
    events = [json.loads(line) for line in traces[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    stamps = [datetime.fromisoformat(event["timestamp"]) for event in events]
    usages = [
        event.get("payload", {}).get("response", {}).get("usage")
        for event in events if event.get("capture_type") == "llm_call"
    ]
    total_tokens = sum(int(usage.get("total_tokens", 0)) for usage in usages if usage)
    return {"runtime_seconds": (max(stamps) - min(stamps)).total_seconds(), "total_tokens": float(total_tokens)}


def summarize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    enriched = []
    for row in payload["results"]:
        enriched.append({**row, **trace_metrics(Path(row["trace_dir"]))})
    table = []
    for arm in ARMS:
        group = [row for row in enriched if row["arm"] == arm]
        noise = [row for row in group if row["case"] == "noise"]
        accuracy = [float((row["case"] == "mechanism") == row["revise_allowed"]) for row in group]
        table.append({
            "harness": "ResearchHarness",
            "variant": NAMES[arm],
            "task_score": mean_sd(accuracy),
            "verified_score": mean_sd(accuracy),
            "false_admission": mean_sd([float(row["revise_allowed"]) for row in noise]),
            "revision_accuracy": mean_sd(accuracy),
            "token_cost": mean_sd([row["total_tokens"] for row in group]),
            "runtime_seconds": mean_sd([row["runtime_seconds"] for row in group]),
            "n_runs": len(group),
        })
    return table


def fmt(value: dict[str, float]) -> str:
    return f"{value['mean']:.3f} ± {value['sd']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    table = summarize(payload)
    out = args.results.parent
    (out / "table7_researchharness.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines = [
        "# Table 7 partial: controlled ResearchHarness transfer",
        "",
        "| Harness | Variant | Task Score ↑ | Verified Score ↑ | False Admission ↓ | Revision Accuracy ↑ | Token Cost ↓ | Runtime (s) ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['harness']} | {row['variant']} | {fmt(row['task_score'])} | {fmt(row['verified_score'])} | "
            f"{fmt(row['false_admission'])} | {fmt(row['revision_accuracy'])} | {fmt(row['token_cost'])} | {fmt(row['runtime_seconds'])} |"
        )
    lines.extend([
        "",
        "Scope: two controlled evidence-admission cases (transient noise and persistent true mechanism), three repeats each. This is not an official ResearchClawBench, AARRI, or AutoResearchEval score.",
        "",
        "AI-Researcher and Claude Code rows remain unrun because no matched three-arm adapter is available locally; they must not be filled from this ResearchHarness result.",
    ])
    (out / "table7_researchharness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
