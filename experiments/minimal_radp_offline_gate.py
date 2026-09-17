#!/usr/bin/env python3
"""Zero-API mechanism gate for the narrow PiEvo/RADP comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import active_principle_discovery_benchmark as apd
import controlled_mbo_policy_controls as controls

PRIMARY_TRUE_CONDITIONS = ("true_mechanism", "noisy_high_fidelity")
STRESS_TRUE_CONDITIONS = ("mixed_noise_mechanism", "weak_true_mechanism")


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def equal_total_metrics(seeds: int) -> dict[str, object]:
    # The old sequential gate is retained as a diagnostic baseline. The live
    # candidate is adaptive_gate, which handles noisy primary observations
    # when repeat and independent verifier values agree above prediction.
    policies = ("source_blind", "adaptive_gate")
    rows = {
        (condition, policy): [
            controls.run_trial(seed, condition, policy, "fixed_total")
            for seed in range(seeds)
        ]
        for condition in controls.CONDITIONS
        for policy in policies
    }
    clean_regret = {
        policy: mean(row.cumulative_regret for row in rows[("clean", policy)])
        for policy in policies
    }
    false_expansion = {
        policy: mean(
            row.false_allow_count
            for condition in controls.NONMECHANISM_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    spurious_regret = {
        policy: mean(
            row.post_trigger_cumulative_regret
            for condition in controls.NONMECHANISM_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    true_accuracy = {
        policy: mean(
            float(row.update_correct)
            for condition in PRIMARY_TRUE_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    stress_true_accuracy = {
        policy: mean(
            float(row.update_correct)
            for condition in STRESS_TRUE_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    checks = {
        "clean_noninferiority": (
            clean_regret["adaptive_gate"]
            <= clean_regret["source_blind"] + 1e-12
        ),
        "spurious_false_expansion_reduced_50pct": (
            false_expansion["adaptive_gate"]
            <= 0.5 * false_expansion["source_blind"]
        ),
        "spurious_post_trigger_regret_lower": (
            spurious_regret["adaptive_gate"]
            < spurious_regret["source_blind"]
        ),
        "primary_true_mechanism_accuracy_noninferior_5pct": (
            true_accuracy["adaptive_gate"]
            >= true_accuracy["source_blind"] - 0.05
        ),
    }
    return {
        "seeds": seeds,
        "budget_mode": "fixed_total",
        "clean_cumulative_regret": clean_regret,
        "nonmechanism_false_allow_count": false_expansion,
        "nonmechanism_post_trigger_regret": spurious_regret,
        "primary_true_mechanism_update_accuracy": true_accuracy,
        "stress_true_mechanism_update_accuracy": stress_true_accuracy,
        "stress_true_mechanism_noninferior_5pct": (
            stress_true_accuracy["adaptive_gate"]
            >= stress_true_accuracy["source_blind"] - 0.05
        ),
        "checks": checks,
    }


def adaptive_candidate_metrics(seeds: int) -> dict[str, object]:
    """Evaluate the zero-API stress-recovery gate without changing the frozen gate.

    This candidate is intentionally diagnostic-only.  Its result is reported
    separately so a better toy stress score cannot silently unlock a paid run
    or alter the preregistered sequential-gate result above.
    """
    policies = ("source_blind", "adaptive_gate")
    rows = {
        (condition, policy): [
            controls.run_trial(seed, condition, policy, "fixed_total")
            for seed in range(seeds)
        ]
        for condition in controls.CONDITIONS
        for policy in policies
    }
    clean_regret = {
        policy: mean(row.cumulative_regret for row in rows[("clean", policy)])
        for policy in policies
    }
    false_expansion = {
        policy: mean(
            row.false_allow_count
            for condition in controls.NONMECHANISM_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    spurious_regret = {
        policy: mean(
            row.post_trigger_cumulative_regret
            for condition in controls.NONMECHANISM_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    true_accuracy = {
        policy: mean(
            float(row.update_correct)
            for condition in PRIMARY_TRUE_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    stress_accuracy = {
        policy: mean(
            float(row.update_correct)
            for condition in STRESS_TRUE_CONDITIONS
            for row in rows[(condition, policy)]
        )
        for policy in policies
    }
    checks = {
        "clean_noninferiority": (
            clean_regret["adaptive_gate"]
            <= clean_regret["source_blind"] + 1e-12
        ),
        "spurious_false_expansion_reduced_50pct": (
            false_expansion["adaptive_gate"]
            <= 0.5 * false_expansion["source_blind"]
        ),
        "spurious_post_trigger_regret_lower": (
            spurious_regret["adaptive_gate"]
            < spurious_regret["source_blind"]
        ),
        "primary_true_mechanism_accuracy_noninferior_5pct": (
            true_accuracy["adaptive_gate"]
            >= true_accuracy["source_blind"] - 0.05
        ),
        "stress_true_mechanism_accuracy_noninferior_5pct": (
            stress_accuracy["adaptive_gate"]
            >= stress_accuracy["source_blind"] - 0.05
        ),
    }
    return {
        "candidate_policy": "adaptive_gate",
        "candidate_is_live": False,
        "anomaly_threshold": controls.ADAPTIVE_ANOMALY_THRESHOLD,
        "verified_gain_threshold": controls.ADAPTIVE_VERIFIED_GAIN,
        "seeds": seeds,
        "budget_mode": "fixed_total",
        "clean_cumulative_regret": clean_regret,
        "nonmechanism_false_allow_count": false_expansion,
        "nonmechanism_post_trigger_regret": spurious_regret,
        "primary_true_mechanism_update_accuracy": true_accuracy,
        "stress_true_mechanism_update_accuracy": stress_accuracy,
        "checks": checks,
        "candidate_stress_gate_passed": all(checks.values()),
    }


def lifecycle_metrics() -> dict[str, object]:
    rows = []
    for seed in (0, 1):
        for condition in apd.CONDITIONS:
            for policy in apd.POLICIES:
                for budget in apd.BUDGETS:
                    rows.append(apd.run_trial(seed, condition, policy, budget))
    checks = apd.proof_checks(rows)
    return {"seeds": [0, 1], "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seeds < 2:
        raise SystemExit("--seeds must be at least 2")
    equal_total = equal_total_metrics(args.seeds)
    adaptive_candidate = adaptive_candidate_metrics(args.seeds)
    lifecycle = lifecycle_metrics()
    checks = {
        **equal_total["checks"],
        "lifecycle_and_backtrack_checks": lifecycle["checks"]["all_checks_pass"],
    }
    passed = all(checks.values())
    stress_passed = equal_total["stress_true_mechanism_noninferior_5pct"]
    result = {
        "schema_version": "minimal_radp_offline_gate.v2",
        "api_calls": 0,
        "equal_total": equal_total,
        "adaptive_candidate": adaptive_candidate,
        "lifecycle": lifecycle,
        "checks": checks,
        "passed": passed,
        "offline_mechanism_gate_passed": passed,
        "controlled_source_confusion_runner_ready": passed,
        "controlled_development_screen_ready": passed,
        "minimal_api_screen_unlocked": False,
        "paid_api_screen_unlocked": False,
        "paid_api_screen_blockers": [
            "explicit user approval for API use has not been supplied",
            "source-confusion is an evaluator-side corruption diagnostic, not an independent physical channel",
            "formal independent and held-out verification remains unavailable",
        ],
        "broad_api_matrix_unlocked": False,
        "stress_gate_passed": stress_passed,
        "adaptive_candidate_stress_gate_passed": adaptive_candidate[
            "candidate_stress_gate_passed"
        ],
        "claim_boundary": (
            "This zero-API gate can support only a controlled mechanism-level advantage. "
            "It does not unlock a paid model screen or establish official-task, paper-level, "
            "or general superiority."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
