#!/usr/bin/env python3
"""Aggregate only fields released with AutoResearchEval optimization rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["harness"], row["model"])].append(row)
    output = []
    for (harness, model), group in sorted(groups.items()):
        scored = [float(r["best_aggregate_improvement"]) for r in group if r.get("best_aggregate_improvement") is not None]
        durations = [float(r["duration_s"]) for r in group]
        output.append({
            "harness": harness,
            "model": model,
            "released_rollouts": len(group),
            "successful_status": sum(r.get("status") == "success" for r in group),
            "runs_with_evaluator_calls": sum(int(r.get("n_evaluations", 0)) > 0 for r in group),
            "critical_failures": sum(r.get("status") != "success" for r in group),
            "mean_best_aggregate_improvement_scored_only": mean(scored),
            "median_best_aggregate_improvement_scored_only": median(scored),
            "mean_runtime_seconds": mean(durations),
            "released_cost": None,
        })
    return output


def fmt(value: float | None) -> str:
    return "not released" if value is None else f"{value:.3f}"


def write_markdown(path: Path, table: list[dict[str, Any]]) -> None:
    lines = [
        "# AutoResearchEval optimization: released-rollout audit",
        "",
        "This table recomputes only fields present in the official 240-rollout release. It is not the proposed five-method Table 6.",
        "",
        "| Harness | Model | Released | Successful | With evaluator calls | Critical failures | Mean best aggregate improvement* | Median* | Mean runtime (s) | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['harness']} | {row['model']} | {row['released_rollouts']} | {row['successful_status']} | "
            f"{row['runs_with_evaluator_calls']} | {row['critical_failures']} | "
            f"{fmt(row['mean_best_aggregate_improvement_scored_only'])} | "
            f"{fmt(row['median_best_aggregate_improvement_scored_only'])} | "
            f"{fmt(row['mean_runtime_seconds'])} | not released |"
        )
    lines.extend([
        "",
        "*Aggregate-improvement summaries condition on rollouts containing at least one evaluator score; unscored runs are not silently imputed as zero.",
        "",
        "The official release contains Claude, Codex, and Gemini harnesses. It contains no AI-Researcher, ResearchHarness, or Triage-PiEvo rollout, and it does not release a unified verified-score, target-attainment, token-cost, or monetary-cost field.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rollouts.read_text(encoding="utf-8").splitlines() if line.strip()]
    table = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "released_rollout_audit.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "released_rollout_audit.md", table)
    manifest = {
        "source": "https://huggingface.co/datasets/PrentisAI/AutoResearchEval",
        "source_dataset_sha": "412fb4908a3a7b98dd50cfa3e5c6cd40750d2d4f",
        "source_file": str(args.rollouts.resolve()),
        "source_file_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        "rows": len(rows),
        "claim_boundary": "Released-rollout audit only; no new AutoResearchEval trajectories were executed.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "groups": len(table), "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
