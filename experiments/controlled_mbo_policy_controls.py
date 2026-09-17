#!/usr/bin/env python3
"""Policy controls for the PiEvo direction-1 source-confusion diagnostic.

This is a deterministic, evaluator-side MBO simulation.  It is intentionally
separate from the LLM PiEvo runs: it provides matched controls and stress
tests for the validation protocol, while the LLM runs establish that the
source-blind failure occurs in the integrated system.

Policies
--------
``source_blind``
    Every detected anomaly is accepted as principle evidence.
``repeat_only``
    Repeat the visible measurement and accept when primary and repeat agree.
``fixed_repeat_independent``
    Always collect both checks and require both to agree with primary.
``sequential_gate``
    Repeat first; only if it agrees, pay for an independent check.
``adaptive_gate``
    Exploratory stress-recovery candidate. It uses a lower anomaly trigger
    for weak effects and can retain a primary spike when the repeat and
    independent channels agree with each other and the independent value is
    above the current-principle prediction. It is a zero-API diagnostic and
    is not wired into the live runner.

The script reports both a fixed-primary-budget view and a fixed-total-call
budget view.  The latter exposes the cost/coverage trade-off instead of
silently giving validation calls for free.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


ROUNDS = 30
TOTAL_CALL_BUDGET = 30
EVENT_ROUNDS = (6, 7, 8)
EVENT_COUNT = len(EVENT_ROUNDS)
MIN_SUPPORTING_EVENTS = 2
ANOMALY_THRESHOLD = 0.85
RESIDUAL_SD = 0.15
REPEAT_TOLERANCE = 0.15
INDEPENDENT_TOLERANCE = 0.15
ADAPTIVE_ANOMALY_THRESHOLD = 0.45
ADAPTIVE_VERIFIED_GAIN = 0.03
POLICIES = (
    "source_blind",
    "repeat_only",
    "fixed_repeat_independent",
    "sequential_gate",
    "adaptive_gate",
)
CONDITIONS = (
    "clean",
    "noise",
    "surrogate_failure",
    "true_mechanism",
    "persistent_measurement_bias",
    "noisy_high_fidelity",
    "weak_true_mechanism",
    "mixed_noise_mechanism",
    "environment_drift",
    "ood_surrogate_failure",
)
TRUE_MECHANISM_CONDITIONS = {
    "true_mechanism",
    "noisy_high_fidelity",
    "weak_true_mechanism",
    "mixed_noise_mechanism",
}
NONMECHANISM_CONDITIONS = set(CONDITIONS) - TRUE_MECHANISM_CONDITIONS - {"clean"}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group: int
    quality: float


@dataclass
class TrialResult:
    seed: int
    budget_mode: str
    condition: str
    policy: str
    primary_rounds: int
    total_calls: int
    detected_event_count: int
    allowed_event_count: int
    false_allow_count: int
    missed_true_event_count: int
    detection_miss_count: int
    principle_updated: bool
    update_correct: Optional[bool]
    update_round: Optional[int]
    principle_persistence_rounds: int
    verification_cost: int
    best_true_reward: float
    oracle_reward: float
    cumulative_regret: float
    post_trigger_cumulative_regret: float
    final_principle: str
    event_decisions: List[Dict[str, object]]


def build_candidates(seed: int) -> List[Candidate]:
    rng = random.Random(seed * 9176 + 31)
    return [Candidate(f"mol_{i:04d}", i % 4, rng.random()) for i in range(512)]


def base_value(candidate: Candidate) -> float:
    return (0.72, 0.60, 0.48, 0.22)[candidate.group] + 0.05 * candidate.quality


def true_value(candidate: Candidate, condition: str) -> float:
    if condition in {"true_mechanism", "noisy_high_fidelity", "mixed_noise_mechanism"} and candidate.group == 3:
        return min(1.0, 0.88 + 0.08 * candidate.quality)
    if condition == "weak_true_mechanism" and candidate.group == 3:
        return min(1.0, 0.34 + 0.04 * candidate.quality)
    return base_value(candidate)


def prediction(candidate: Candidate, principle: str) -> float:
    if principle == "P1" and candidate.group == 3:
        return min(1.0, 0.88 + 0.08 * candidate.quality)
    return base_value(candidate)


def event_primary(candidate: Candidate, condition: str, rng: random.Random) -> float:
    """Visible primary value for a forced diagnostic event."""
    if condition == "weak_true_mechanism":
        return true_value(candidate, condition) + rng.gauss(0.0, 0.01)
    if condition == "mixed_noise_mechanism":
        return 1.08 + rng.gauss(0.0, 0.01)
    if condition == "true_mechanism":
        return 0.94 + rng.gauss(0.0, 0.01)
    if condition == "noisy_high_fidelity":
        return 0.94 + rng.gauss(0.0, 0.01)
    # A matched one-shot visible spike for source-confusion conditions.
    return 0.94 + rng.gauss(0.0, 0.01)


def ordinary_observation(candidate: Candidate, condition: str, principle: str, rng: random.Random) -> float:
    value = true_value(candidate, condition)
    if condition in {"surrogate_failure", "ood_surrogate_failure", "persistent_measurement_bias"} and candidate.group == 3:
        value = min(1.0, value + 0.65)
    return value + rng.gauss(0.0, 0.01)


def verification_values(
    candidate: Candidate,
    condition: str,
    primary: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Return hidden repeat and independent observations for one event."""
    baseline = base_value(candidate)
    latent = true_value(candidate, condition)
    if condition == "noise":
        return baseline, baseline
    if condition in {"surrogate_failure", "ood_surrogate_failure", "persistent_measurement_bias"}:
        return primary, baseline
    if condition in {"true_mechanism", "weak_true_mechanism"}:
        return primary, latent
    if condition == "noisy_high_fidelity":
        return primary, latent + rng.gauss(0.0, 0.10)
    if condition == "mixed_noise_mechanism":
        return latent, latent + rng.gauss(0.0, 0.01)
    if condition == "environment_drift":
        # Every measurement channel shares the same drift.  No observational
        # gate can distinguish this from a mechanism without a control arm.
        return primary, primary + rng.gauss(0.0, 0.01)
    return baseline, baseline


def surprisal(observed: float, expected: float) -> float:
    z = abs(observed - expected) / RESIDUAL_SD
    return 1.0 - math.exp(-math.sqrt(z * z))


def choose_candidate(candidates: Sequence[Candidate], tested: set[str], principle: str, rng: random.Random) -> Candidate:
    pool = [candidate for candidate in candidates if candidate.candidate_id not in tested]
    if not pool:
        pool = list(candidates)
    if rng.random() < 0.10:
        return rng.choice(pool)
    return max(pool, key=lambda candidate: prediction(candidate, principle))


def gate_event(
    policy: str,
    primary: float,
    repeat: float,
    independent: float,
    *,
    expected: float | None = None,
) -> tuple[bool, int, str]:
    repeat_ok = abs(primary - repeat) <= REPEAT_TOLERANCE
    independent_ok = abs(primary - independent) <= INDEPENDENT_TOLERANCE
    if policy == "source_blind":
        return True, 0, "source_blind_allow"
    if policy == "repeat_only":
        return repeat_ok, 1, "repeat_agree" if repeat_ok else "repeat_disagree"
    if policy == "fixed_repeat_independent":
        allowed = repeat_ok and independent_ok
        return allowed, 2, "both_agree" if allowed else "fixed_check_disagree"
    if policy == "adaptive_gate":
        if expected is None or not math.isfinite(float(expected)):
            raise ValueError("adaptive gate requires a finite expected outcome")
        verifier_consensus = abs(repeat - independent) <= REPEAT_TOLERANCE
        verified_gain = float(independent) - float(expected)
        if repeat_ok and independent_ok:
            return True, 2, "adaptive_all_channels_agree"
        if verifier_consensus and verified_gain >= ADAPTIVE_VERIFIED_GAIN:
            return True, 2, "adaptive_primary_spike_verified_consensus"
        return False, 2, "adaptive_verifier_disagreement_or_weak_gain"
    # Sequential gate: independent verification is not paid for after a
    # repeat mismatch.
    if not repeat_ok:
        return False, 1, "repeat_disagree"
    return independent_ok, 2, "sequential_both_agree" if independent_ok else "independent_disagree"


def run_trial(seed: int, condition: str, policy: str, budget_mode: str) -> TrialResult:
    if policy not in POLICIES:
        raise ValueError(policy)
    if budget_mode not in {"fixed_primary", "fixed_total"}:
        raise ValueError(budget_mode)
    candidates = build_candidates(seed)
    event_rng = random.Random(seed * 101 + 7)
    event_candidates = event_rng.sample([c for c in candidates if c.group == 3], k=EVENT_COUNT)
    event_by_round = dict(zip(EVENT_ROUNDS, event_candidates))
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    # All policies in a (seed, condition) cell must see the same candidate and
    # observation stream. Policy-specific RNG offsets make a nominally paired
    # comparison confound policy effects with different random trajectories.
    rng = random.Random(seed * 1009 + (CONDITIONS.index(condition) + 1) * 100003)
    verify_rng = random.Random(seed * 9176 + CONDITIONS.index(condition) * 113 + 17)
    tested: set[str] = set()
    principle = "P0"
    selected_values: List[float] = []
    event_decisions: List[Dict[str, object]] = []
    detected = 0
    allowed = 0
    false_allow = 0
    missed_true = 0
    detection_miss = 0
    verification_cost = 0
    total_calls = 0
    update_round: Optional[int] = None
    max_primary = ROUNDS if budget_mode == "fixed_primary" else TOTAL_CALL_BUDGET
    oracle = max(true_value(candidate, condition) for candidate in candidates)

    t = 0
    while t < max_primary and total_calls < (10**9 if budget_mode == "fixed_primary" else TOTAL_CALL_BUDGET):
        total_calls += 1
        if condition != "clean" and t in event_by_round:
            candidate = event_by_round[t]
        else:
            candidate = choose_candidate(candidates, tested, principle, rng)
        tested.add(candidate.candidate_id)
        y_true = true_value(candidate, condition)
        if condition != "clean" and t in event_by_round:
            primary = event_primary(candidate, condition, rng)
            event = True
        else:
            primary = ordinary_observation(candidate, condition, principle, rng)
            event = False
        selected_values.append(y_true)
        expected = prediction(candidate, principle)
        score = surprisal(primary, expected)

        if event:
            anomaly_threshold = (
                ADAPTIVE_ANOMALY_THRESHOLD
                if policy == "adaptive_gate"
                else ANOMALY_THRESHOLD
            )
            if score > anomaly_threshold:
                detected += 1
                repeat, independent = verification_values(candidate, condition, primary, verify_rng)
                is_allowed, cost, reason = gate_event(
                    policy,
                    primary,
                    repeat,
                    independent,
                    expected=expected,
                )
                if budget_mode == "fixed_total" and total_calls + cost > TOTAL_CALL_BUDGET:
                    is_allowed, cost, reason = False, 0, "budget_exhausted_before_verification"
                total_calls += cost
                verification_cost += cost
                allowed += int(is_allowed)
                if condition in NONMECHANISM_CONDITIONS:
                    false_allow += int(is_allowed)
                if condition in TRUE_MECHANISM_CONDITIONS:
                    missed_true += int(not is_allowed)
                event_decisions.append({
                    "round": t,
                    "score": score,
                    "threshold": anomaly_threshold,
                    "primary": primary,
                    "repeat": repeat,
                    "independent": independent,
                    "allowed": is_allowed,
                    "cost": cost,
                    "reason": reason,
                })
            else:
                detection_miss += 1
                if condition in TRUE_MECHANISM_CONDITIONS:
                    missed_true += 1
                event_decisions.append({
                    "round": t,
                    "score": score,
                    "threshold": anomaly_threshold,
                    "allowed": False,
                    "cost": 0,
                    "reason": "below_anomaly_threshold",
                })
        t += 1

        # Match the live controller: acceptance requires two distinct pieces
        # of qualifying evidence. Requiring 3/3 here made the evaluator more
        # conservative than the integrated algorithm and inflated misses.
        if (
            t - 1 == EVENT_ROUNDS[-1]
            and detected == EVENT_COUNT
            and allowed >= MIN_SUPPORTING_EVENTS
        ):
            principle = "P1"
            update_round = t - 1

    primary_rounds = len(selected_values)
    updated = update_round is not None
    if condition == "clean":
        update_correct: Optional[bool] = not updated
    elif condition in TRUE_MECHANISM_CONDITIONS:
        update_correct = updated
    else:
        update_correct = not updated
    persistence = max(0, primary_rounds - 1 - update_round) if update_round is not None else 0
    regrets = [oracle - value for value in selected_values]
    post_trigger = sum(regrets[EVENT_ROUNDS[-1] + 1 :])
    return TrialResult(
        seed=seed,
        budget_mode=budget_mode,
        condition=condition,
        policy=policy,
        primary_rounds=primary_rounds,
        total_calls=total_calls,
        detected_event_count=detected,
        allowed_event_count=allowed,
        false_allow_count=false_allow,
        missed_true_event_count=missed_true,
        detection_miss_count=detection_miss,
        principle_updated=updated,
        update_correct=update_correct,
        update_round=update_round,
        principle_persistence_rounds=persistence,
        verification_cost=verification_cost,
        best_true_reward=max(selected_values) if selected_values else 0.0,
        oracle_reward=oracle,
        cumulative_regret=sum(regrets),
        post_trigger_cumulative_regret=post_trigger,
        final_principle=principle,
        event_decisions=event_decisions,
    )


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def summarize(rows: Sequence[TrialResult]) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, Dict[str, List[TrialResult]]]] = {}
    for row in rows:
        grouped.setdefault(row.budget_mode, {}).setdefault(row.condition, {}).setdefault(row.policy, []).append(row)
    result: Dict[str, object] = {}
    for budget_mode, conditions in grouped.items():
        result[budget_mode] = {}
        for condition, policies in conditions.items():
            result[budget_mode][condition] = {}
            for policy, trials in policies.items():
                correct = [r.update_correct for r in trials if r.update_correct is not None]
                result[budget_mode][condition][policy] = {
                    "n": len(trials),
                    "update_rate": mean(float(r.principle_updated) for r in trials),
                    "correct_decision_rate": mean(float(x) for x in correct) if correct else None,
                    "mean_detected_event_count": mean(r.detected_event_count for r in trials),
                    "mean_allowed_event_count": mean(r.allowed_event_count for r in trials),
                    "mean_false_allow_count": mean(r.false_allow_count for r in trials),
                    "mean_missed_true_event_count": mean(r.missed_true_event_count for r in trials),
                    "mean_detection_miss_count": mean(r.detection_miss_count for r in trials),
                    "mean_verification_cost": mean(r.verification_cost for r in trials),
                    "mean_primary_rounds": mean(r.primary_rounds for r in trials),
                    "mean_total_calls": mean(r.total_calls for r in trials),
                    "mean_principle_persistence_rounds": mean(r.principle_persistence_rounds for r in trials),
                    "mean_best_true_reward": mean(r.best_true_reward for r in trials),
                    "mean_cumulative_regret": mean(r.cumulative_regret for r in trials),
                    "mean_post_trigger_cumulative_regret": mean(r.post_trigger_cumulative_regret for r in trials),
                }
            blind = {r.seed: r for r in policies.get("source_blind", [])}
            if blind:
                result[budget_mode][condition]["paired_delta_vs_source_blind"] = {
                    policy: {
                        "mean_cumulative_regret_delta": mean(
                            trial.cumulative_regret - blind[trial.seed].cumulative_regret
                            for trial in policies[policy]
                            if trial.seed in blind
                        )
                    }
                    for policy in policies
                    if policy != "source_blind"
                }
    return result


def write_outputs(rows: Sequence[TrialResult], output_dir: Path, seeds: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serialised = [asdict(row) for row in rows]
    (output_dir / "trial_results.json").write_text(json.dumps(serialised, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({
            "description": "Evaluator-side policy controls and stress tests; not an official PiEvo benchmark.",
            "seeds": seeds,
            "rounds": ROUNDS,
            "event_rounds": EVENT_ROUNDS,
            "policies": POLICIES,
            "conditions": CONDITIONS,
            "fixed_total_call_budget": TOTAL_CALL_BUDGET,
            "summary": summarize(rows),
        }, indent=2), encoding="utf-8"
    )
    fields = [
        "seed", "budget_mode", "condition", "policy", "primary_rounds", "total_calls",
        "detected_event_count", "allowed_event_count", "false_allow_count",
        "missed_true_event_count", "detection_miss_count", "principle_updated",
        "update_correct", "update_round", "principle_persistence_rounds", "verification_cost",
        "best_true_reward", "oracle_reward", "cumulative_regret",
        "post_trigger_cumulative_regret", "final_principle",
    ]
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in serialised:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: List[TrialResult] = []
    for budget_mode in ("fixed_primary", "fixed_total"):
        for seed in range(args.seeds):
            for condition in CONDITIONS:
                for policy in POLICIES:
                    rows.append(run_trial(seed, condition, policy, budget_mode))
    write_outputs(rows, args.output_dir, args.seeds)
    print(json.dumps({"seeds": args.seeds, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
