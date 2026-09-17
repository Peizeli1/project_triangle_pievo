#!/usr/bin/env python3
"""Summarize the frozen AARRI matrix into the three paper-facing tables."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tomllib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments/results/aarri_formal_matrix_20260903"
ABLATION = ROOT / "experiments/results/evidence_admission_ablation_20260901/table8.json"
ARMS = ("base", "self_review", "stateful_triage")
DISPLAY = {
    "base": "Terminus-2",
    "self_review": "Terminus-2 + Self-Review",
    "stateful_triage": "Terminus-2 + Stateful Triage",
}
SEEDS = (0, 1, 2)


def mean_sd(values: list[float]) -> str:
    if not values:
        return "TBD"
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {sd:.3f}"


def elapsed_seconds(row: dict) -> float | None:
    try:
        start = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(row["finished_at"].replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def task_metadata() -> dict[str, dict]:
    out = {}
    root = ROOT / "experiments/external_harness_pilot/AARRI-bench/tasks"
    for path in root.glob("*/task.toml"):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        out[path.parent.name] = data.get("metadata", {})
    return out


def load_trials() -> list[dict]:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            job_root = RESULT_ROOT / f"{arm}_seed{seed}"
            candidates = defaultdict(list)
            for path in sorted(job_root.glob("**/result.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                task = data.get("task_name")
                trial = data.get("trial_name")
                if not task or not trial:
                    continue
                verifier = data.get("verifier_result") or {}
                reward = (verifier.get("rewards") or {}).get("reward")
                agent = data.get("agent_result") or {}
                candidates[task].append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "task": task,
                        "reward": float(reward) if reward is not None else None,
                        "exception": data.get("exception_info"),
                        "input_tokens": agent.get("n_input_tokens"),
                        "output_tokens": agent.get("n_output_tokens"),
                        "cost_usd": agent.get("cost_usd"),
                        "runtime_seconds": elapsed_seconds(data),
                        "result_path": str(path.relative_to(ROOT)),
                        "finished_at": data.get("finished_at") or "",
                    }
                )
            for task_rows in candidates.values():
                task_rows.sort(
                    key=lambda row: (
                        row["reward"] is not None and not row["exception"],
                        row["finished_at"],
                    )
                )
                rows.append(task_rows[-1])
    return rows


def gate_counts() -> Counter:
    counts = Counter()
    for path in RESULT_ROOT.glob("stateful_triage_seed*/**/agent/triage_evidence_state.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        decisions = []
        for claim in state.get("claims", {}).values():
            decisions.extend(claim.get("decision_history", []))
        if decisions:
            counts["engaged_trials"] += 1
        for decision in decisions:
            counts[str(decision.get("action", "UNKNOWN"))] += 1
    return counts


def exact_paired_sign_test(a: list[float], b: list[float]) -> dict:
    base_wins = sum(x > y for x, y in zip(a, b))
    triage_wins = sum(y > x for x, y in zip(a, b))
    discordant = base_wins + triage_wins
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(base_wins, triage_wins) + 1)) / (2**discordant)
        p = min(1.0, 2 * tail)
    return {"base_wins": base_wins, "triage_wins": triage_wins, "discordant": discordant, "p_two_sided": p}


def summarize(rows: list[dict]) -> dict:
    meta = task_metadata()
    by_key = {(r["arm"], r["seed"], r["task"]): r for r in rows}
    overall = []
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        seed_means = []
        for seed in SEEDS:
            valid = [r["reward"] for r in arm_rows if r["seed"] == seed and r["reward"] is not None and not r["exception"]]
            if len(valid) == 82:
                seed_means.append(statistics.fmean(valid))
        valid_rows = [r for r in arm_rows if r["reward"] is not None and not r["exception"]]
        overall.append(
            {
                "arm": arm,
                "display": DISPLAY[arm],
                "score_by_seed": seed_means,
                "score_mean_sd": mean_sd(seed_means),
                "valid_runs": len(valid_rows),
                "expected_runs": 246,
                "critical_failures": sum(bool(r["exception"]) for r in arm_rows),
                "mean_total_tokens": statistics.fmean(
                    (r["input_tokens"] or 0) + (r["output_tokens"] or 0)
                    for r in valid_rows
                ) if valid_rows else None,
                "mean_runtime_seconds": statistics.fmean(
                    r["runtime_seconds"] for r in valid_rows if r["runtime_seconds"] is not None
                ) if valid_rows else None,
                "mean_cost_usd": statistics.fmean(
                    r["cost_usd"] for r in valid_rows if r["cost_usd"] is not None
                ) if any(r["cost_usd"] is not None for r in valid_rows) else None,
            }
        )

    categories = sorted({m.get("category", "unknown") for m in meta.values()})
    difficulties = sorted({m.get("difficulty", "unknown") for m in meta.values()})
    breakdown = []
    for arm in ARMS:
        row = {"arm": arm, "display": DISPLAY[arm], "category": {}, "difficulty": {}}
        for field, values in (("category", categories), ("difficulty", difficulties)):
            for value in values:
                seed_scores = []
                tasks = [task for task, m in meta.items() if m.get(field, "unknown") == value]
                for seed in SEEDS:
                    scores = [by_key.get((arm, seed, task), {}).get("reward") for task in tasks]
                    if scores and all(score is not None for score in scores):
                        seed_scores.append(statistics.fmean(scores))
                row[field][value] = {"n_tasks": len(tasks), "scores": seed_scores, "mean_sd": mean_sd(seed_scores)}
        breakdown.append(row)

    paired = {}
    for competitor in ("base", "self_review"):
        a, b = [], []
        for seed in SEEDS:
            for task in sorted(meta):
                left = by_key.get((competitor, seed, task), {}).get("reward")
                right = by_key.get(("stateful_triage", seed, task), {}).get("reward")
                if left is not None and right is not None:
                    a.append(left)
                    b.append(right)
        paired[competitor] = exact_paired_sign_test(a, b)

    return {
        "protocol": json.loads((RESULT_ROOT / "frozen_protocol.json").read_text(encoding="utf-8")),
        "overall": overall,
        "breakdown": breakdown,
        "paired_exact_sign_tests": paired,
        "gate_counts": dict(gate_counts()),
        "observed_trials": len(rows),
        "expected_trials": 738,
        "complete": len(rows) == 738 and all(item["valid_runs"] == 246 for item in overall),
    }


def render(summary: dict) -> str:
    lines = [
        "# AARRI formal matrix: Tables 6–8",
        "",
        f"Status: {'complete' if summary['complete'] else 'running'} ({summary['observed_trials']}/{summary['expected_trials']} terminal trials).",
        "",
        "## Table 6: Overall AARRI performance",
        "",
        "| Method | Verified score ↑ | Valid runs | Critical failures ↓ | Tokens / task ↓ | Runtime / task (s) ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["overall"]:
        tokens = f"{row['mean_total_tokens']:.1f}" if row["mean_total_tokens"] is not None else "TBD"
        runtime = f"{row['mean_runtime_seconds']:.1f}" if row["mean_runtime_seconds"] is not None else "TBD"
        lines.append(f"| {row['display']} | {row['score_mean_sd']} | {row['valid_runs']}/{row['expected_runs']} | {row['critical_failures']} | {tokens} | {runtime} |")

    lines.extend(["", "## Table 7: AARRI category breakdown", ""])
    categories = sorted(summary["breakdown"][0]["category"])
    lines.append("| Method | " + " | ".join(f"{c} ↑" for c in categories) + " |")
    lines.append("|---|" + "---:|" * len(categories))
    for row in summary["breakdown"]:
        lines.append("| " + row["display"] + " | " + " | ".join(row["category"][c]["mean_sd"] for c in categories) + " |")

    lines.extend(["", "### Difficulty breakdown", ""])
    difficulties = sorted(summary["breakdown"][0]["difficulty"])
    lines.append("| Method | " + " | ".join(f"{d} ↑" for d in difficulties) + " |")
    lines.append("|---|" + "---:|" * len(difficulties))
    for row in summary["breakdown"]:
        lines.append("| " + row["display"] + " | " + " | ".join(row["difficulty"][d]["mean_sd"] for d in difficulties) + " |")

    lines.extend(["", "## Table 8: Evidence-admission ablation", ""])
    if ABLATION.is_file():
        table8_md = ABLATION.with_suffix(".md")
        lines.append(table8_md.read_text(encoding="utf-8").strip())
    else:
        lines.append("TBD: completed ablation artifact not found.")

    lines.extend([
        "",
        "## Stateful Triage audit",
        "",
        "```json",
        json.dumps({"gate_counts": summary["gate_counts"], "paired_exact_sign_tests": summary["paired_exact_sign_tests"]}, indent=2),
        "```",
        "",
        "Scores are official AARRI verifier rewards. Smoke-test runs are excluded. Cost remains unavailable when the relay does not return provider pricing metadata.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_trials()
    summary = summarize(rows)
    (RESULT_ROOT / "formal_tables.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULT_ROOT / "formal_tables.md").write_text(render(summary), encoding="utf-8")
    print(json.dumps({"complete": summary["complete"], "observed": summary["observed_trials"], "expected": summary["expected_trials"]}))
    if args.require_complete and not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
