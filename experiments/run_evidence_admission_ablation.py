#!/usr/bin/env python3
"""Run the evidence-admission ablations used by Table 8.

This script reuses the frozen hidden-source environment from
``active_principle_discovery_benchmark.py`` but gives the manuscript's gate
parameter k its literal meaning: the number of independent held-out support
records required before a principle is accepted.  Full Triage uses k=2.

The study is evaluator-side and deterministic.  It is not an AutoResearchEval
or chemistry leaderboard result; its purpose is causal attribution of the
evidence-admission components under matched hidden-source interventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import active_principle_discovery_benchmark as env


VARIANTS: dict[str, dict[str, Any]] = {
    "full_triage": {"k": 2, "heldout": True, "reliability": True, "backtrack": True, "independent": True},
    "k_1": {"k": 1, "heldout": True, "reliability": True, "backtrack": True, "independent": True},
    "k_3": {"k": 3, "heldout": True, "reliability": True, "backtrack": True, "independent": True},
    "without_heldout": {"k": 2, "heldout": False, "reliability": True, "backtrack": True, "independent": True},
    "without_channel_reliability": {"k": 2, "heldout": True, "reliability": False, "backtrack": True, "independent": True},
    "without_backtracking": {"k": 2, "heldout": True, "reliability": True, "backtrack": False, "independent": True},
    "same_source_as_independent": {"k": 2, "heldout": True, "reliability": True, "backtrack": True, "independent": False},
}

DISPLAY_NAMES = {
    "full_triage": "Full Triage (k=2)",
    "k_1": "k=1",
    "k_3": "k=3",
    "without_heldout": "w/o held-out verification",
    "without_channel_reliability": "w/o channel reliability",
    "without_backtracking": "w/o backtracking",
    "same_source_as_independent": "Same-source repeat = independent",
}


@dataclass
class Result:
    seed: int
    condition: str
    variant: str
    budget: int
    accepted_round: int | None
    final_principle: str
    false_revision: bool
    true_mechanism_recalled: bool | None
    final_state_correct: bool
    retraction_correct: bool | None
    verification_cost: int
    supports: int
    contradictions: int
    backtracks: int
    events: list[dict[str, Any]]


def _trust(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _verify_event(
    *,
    candidate: env.Candidate,
    condition: str,
    observed: float,
    round_index: int,
    candidates: list[env.Candidate],
    rng: random.Random,
    config: dict[str, Any],
    channel_state: dict[str, list[float]],
) -> dict[str, Any]:
    """Verify one anomalous record without exposing the hidden condition."""
    channels = env.verify_channel(candidate, condition, observed, round_index, rng)
    repeat = channels["repeat"]
    independent = channels["independent"] if config["independent"] else repeat
    repeat_ok = abs(observed - repeat) <= env.REPEAT_TOLERANCE
    independent_ok = abs(observed - independent) <= env.INDEPENDENT_TOLERANCE
    trace = ["repeat", "independent" if config["independent"] else "same_source_repeat"]
    cost = 2

    # Reliability is learned only from observable channel agreement.  It does
    # not use the hidden source label.  A low-trust channel requires held-out
    # corroboration but can never authorize a revision by itself.
    if config["reliability"]:
        for name, ok in (("repeat", repeat_ok), ("independent", independent_ok)):
            channel_state[name][0 if ok else 1] += 1.0
        channel_trust = {
            name: _trust(*masses) for name, masses in channel_state.items()
        }
    else:
        channel_trust = {"repeat": 0.5, "independent": 0.5}

    if not repeat_ok or not independent_ok:
        return {
            "support": False,
            "contradiction": True,
            "cost": cost,
            "trace": trace,
            "decision": "channel_disagreement",
            "channel_trust": channel_trust,
            "heldout_gain": None,
        }

    if not config["heldout"]:
        return {
            "support": True,
            "contradiction": False,
            "cost": cost,
            "trace": trace,
            "decision": "admit_without_heldout",
            "channel_trust": channel_trust,
            "heldout_gain": None,
        }

    gain = env.heldout_gain(candidates, condition, round_index)
    trace.append("heldout_probe")
    cost += env.HELDOUT_PROBE_COST
    support = gain >= env.HELDOUT_GAIN_THRESHOLD
    return {
        "support": support,
        "contradiction": not support,
        "cost": cost,
        "trace": trace,
        "decision": "heldout_support" if support else "heldout_contradiction",
        "channel_trust": channel_trust,
        "heldout_gain": gain,
    }


def run_trial(seed: int, condition: str, variant: str, budget: int = 48) -> Result:
    config = VARIANTS[variant]
    candidates = env.build_candidates(seed)
    forced = {
        round_index: env.event_candidate(candidates, condition, index)
        for index, round_index in enumerate(env.EVENT_ROUNDS)
    }
    rng = random.Random(1009 * seed + 100003 * (env.CONDITIONS.index(condition) + 1) + 271)
    verify_rng = random.Random(9176 * seed + 113 * env.CONDITIONS.index(condition) + 29)
    tested: set[str] = set()
    principle = "P0"
    accepted_round: int | None = None
    supports = contradictions = verification_cost = backtracks = 0
    events: list[dict[str, Any]] = []
    channel_state = {"repeat": [1.0, 1.0], "independent": [1.0, 1.0]}

    for t in range(budget):
        candidate = forced[t] if t in forced else env.choose_candidate(candidates, tested, principle, rng)
        tested.add(candidate.candidate_id)
        is_forced_event = t in forced and condition != "clean"
        observed = env.primary_observation(candidate, condition, t, is_forced_event, rng)
        score = env.surprisal(observed, env.model_expected(candidate, principle))

        if is_forced_event and score > env.ANOMALY_THRESHOLD:
            decision = _verify_event(
                candidate=candidate,
                condition=condition,
                observed=observed,
                round_index=t,
                candidates=candidates,
                rng=verify_rng,
                config=config,
                channel_state=channel_state,
            )
            verification_cost += decision["cost"]
            supports += int(decision["support"])
            contradictions += int(decision["contradiction"])
            events.append({"round": t, "candidate_id": candidate.candidate_id, "surprisal": score, **decision})
            if principle == "P0" and supports >= config["k"]:
                principle = "P1"
                accepted_round = t

        if principle == "P1" and t in env.BACKTRACK_CHECK_ROUNDS:
            gain = env.heldout_gain(candidates, condition, t)
            verification_cost += env.HELDOUT_PROBE_COST
            if gain < env.HELDOUT_GAIN_THRESHOLD and config["backtrack"]:
                principle = "P0"
                backtracks += 1

    true_recall = accepted_round is not None if condition in env.TRUE_INITIAL_CONDITIONS else None
    retraction = (principle == "P0" and backtracks > 0) if condition == "mechanism_reversal" else None
    return Result(
        seed=seed,
        condition=condition,
        variant=variant,
        budget=budget,
        accepted_round=accepted_round,
        final_principle=principle,
        false_revision=condition in env.NONMECHANISM_CONDITIONS and accepted_round is not None,
        true_mechanism_recalled=true_recall,
        final_state_correct=bool(env.final_correct(condition, principle)),
        retraction_correct=retraction,
        verification_cost=verification_cost,
        supports=supports,
        contradictions=contradictions,
        backtracks=backtracks,
        events=events,
    )


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(rows: list[Result]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for variant in VARIANTS:
        group = [r for r in rows if r.variant == variant]
        false_rows = [r for r in group if r.condition in {"noise", "surrogate_failure", "local_artifact"}]
        mechanism_rows = [r for r in group if r.condition == "true_mechanism"]
        reversal_rows = [r for r in group if r.condition == "mechanism_reversal"]
        table.append({
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
            "false_revision": _mean_sd([float(r.false_revision) for r in false_rows]),
            "true_mechanism_recall": _mean_sd([float(r.true_mechanism_recalled) for r in mechanism_rows]),
            "verified_state_accuracy": _mean_sd([float(r.final_state_correct) for r in group]),
            "retraction_accuracy": _mean_sd([float(r.retraction_correct) for r in reversal_rows]),
            "verification_cost": _mean_sd([float(r.verification_cost) for r in group]),
            "n_trials": len(group),
        })
    return table


def _fmt(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} ± {metric['sd']:.3f}"


def write_markdown(path: Path, table: list[dict[str, Any]]) -> None:
    lines = [
        "# Table 8: Evidence-admission ablation",
        "",
        "All rates and costs are mean ± sample SD over the frozen two-seed hidden-source intervention suite at budget 48. Full Triage uses k=2.",
        "",
        "| Variant | False Revision ↓ | True Mechanism Recall ↑ | Verified State Accuracy ↑ | Retraction Accuracy ↑ | Verification Cost ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['display_name']} | {_fmt(row['false_revision'])} | {_fmt(row['true_mechanism_recall'])} | "
            f"{_fmt(row['verified_state_accuracy'])} | {_fmt(row['retraction_accuracy'])} | {_fmt(row['verification_cost'])} |"
        )
    lines.extend([
        "",
        "Metric scope: false revision is evaluated on noise, surrogate-failure, and local-artifact trials; true-mechanism recall on the persistent true-mechanism condition; retraction accuracy on mechanism reversal; verified state accuracy and verification cost on all six conditions.",
        "",
        "This is a controlled causal benchmark, not an AutoResearchEval or chemistry leaderboard result.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def proof_checks(table: list[dict[str, Any]]) -> dict[str, bool]:
    by_variant = {row["variant"]: row for row in table}
    checks = {
        "full_blocks_false_revision": by_variant["full_triage"]["false_revision"]["mean"] == 0.0,
        "full_recalls_true_mechanism": by_variant["full_triage"]["true_mechanism_recall"]["mean"] == 1.0,
        "full_retracts_reversal": by_variant["full_triage"]["retraction_accuracy"]["mean"] == 1.0,
        "without_heldout_false_revises": by_variant["without_heldout"]["false_revision"]["mean"] > 0.0,
        "without_backtracking_fails_retraction": by_variant["without_backtracking"]["retraction_accuracy"]["mean"] == 0.0,
        "k_cost_is_monotone": (
            by_variant["k_1"]["verification_cost"]["mean"]
            < by_variant["full_triage"]["verification_cost"]["mean"]
            < by_variant["k_3"]["verification_cost"]["mean"]
        ),
    }
    checks["all_checks_pass"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(env.SEEDS))
    parser.add_argument("--budget", type=int, default=48)
    args = parser.parse_args()
    if sorted(args.seeds) != list(env.SEEDS):
        raise SystemExit("The matched benchmark is frozen to seeds 0 and 1.")

    rows = [
        run_trial(seed, condition, variant, args.budget)
        for variant in VARIANTS
        for condition in env.CONDITIONS
        for seed in args.seeds
    ]
    table = summarize(rows)
    checks = proof_checks(table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = [asdict(row) for row in rows]
    (args.output_dir / "trial_results.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with (args.output_dir / "trial_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in asdict(rows[0]) if key != "events"])
        writer.writeheader()
        writer.writerows([{k: v for k, v in asdict(row).items() if k != "events"} for row in rows])
    (args.output_dir / "table8.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "table8.md", table)
    manifest = {
        "protocol": "evidence-admission-ablation-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_script_sha256": hashlib.sha256(Path(env.__file__).read_bytes()).hexdigest(),
        "seeds": args.seeds,
        "budget": args.budget,
        "conditions": list(env.CONDITIONS),
        "variants": VARIANTS,
        "full_k": 2,
        "proof_checks": checks,
        "claim_boundary": "Controlled hidden-source causal ablation; not an external-harness or task-performance leaderboard.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"trials": len(rows), "output_dir": str(args.output_dir.resolve()), "proof_checks": checks, "table": table}, indent=2))
    if not checks["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
