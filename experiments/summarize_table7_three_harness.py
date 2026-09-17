#!/usr/bin/env python3
"""Build the completed three-harness controlled transfer table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ARM_NAMES = {"base": "Base", "vanilla": "Base", "self_review": "+ Self-Review", "triage": "+ Triage", "gate": "+ Triage"}


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def aggregate_raw(harness: str, rows: list[dict[str, Any]], token_fn, runtime_fn) -> list[dict[str, Any]]:
    output = []
    for arm in ("base", "self_review", "triage"):
        aliases = {"base": {"base", "vanilla"}, "self_review": {"self_review"}, "triage": {"triage", "gate"}}[arm]
        group = [row for row in rows if row["arm"] in aliases]
        expected = [row["case"] == "mechanism" for row in group]
        accuracy = [float(bool(row["revise_allowed"]) == truth) for row, truth in zip(group, expected)]
        noise = [row for row in group if row["case"] == "noise"]
        output.append({
            "harness": harness,
            "variant": ARM_NAMES[arm],
            "task_score": mean_sd(accuracy),
            "verified_score": mean_sd(accuracy),
            "false_admission": mean_sd([float(row["revise_allowed"]) for row in noise]),
            "revision_accuracy": mean_sd(accuracy),
            "token_cost": mean_sd([float(token_fn(row)) for row in group]),
            "runtime_seconds": mean_sd([float(runtime_fn(row)) for row in group]),
            "n_runs": len(group),
        })
    return output


def claude_tokens(row: dict[str, Any]) -> int:
    total = 0
    for usage in (row.get("model_usage") or {}).values():
        total += int(usage.get("inputTokens") or 0)
        total += int(usage.get("outputTokens") or 0)
        total += int(usage.get("cacheReadInputTokens") or 0)
        total += int(usage.get("cacheCreationInputTokens") or 0)
    return total


def fmt(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} ± {metric['sd']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--researchharness", type=Path, required=True)
    parser.add_argument("--claude", type=Path, required=True)
    parser.add_argument("--ai-researcher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    research = json.loads(args.researchharness.read_text(encoding="utf-8"))
    claude = json.loads(args.claude.read_text(encoding="utf-8"))["results"]
    ai = json.loads(args.ai_researcher.read_text(encoding="utf-8"))["results"]
    table = list(research)
    table.extend(aggregate_raw("AI-Researcher", ai, lambda row: row["total_tokens"], lambda row: row["runtime_seconds"]))
    table.extend(aggregate_raw("Claude Code", claude, claude_tokens, lambda row: row["duration_ms"] / 1000.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "table7.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines = [
        "# Table 7: Triage plug-in across autonomous research harnesses",
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
        "Protocol: Qwen3-235B-A22B-Instruct-2507 through the same relay; two matched evidence-admission cases (transient noise and persistent true mechanism), three repeats per case and variant. Task/verified/revision scores are decision accuracy against the evaluator-hidden source condition. False admission is measured on transient-noise trials. Token cost counts model tokens reported by each harness.",
        "",
        "Scope: controlled plug-in transfer, not an official ResearchClawBench, AARRI, or AutoResearchEval leaderboard result.",
    ])
    (args.output_dir / "table7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
