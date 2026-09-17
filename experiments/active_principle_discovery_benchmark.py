#!/usr/bin/env python3
"""Controlled active principle-discovery benchmark for PiEvo direction 1.

This benchmark is an evaluator-side companion to the LLM-in-the-loop MBO
integration.  It keeps the hidden source/mechanism labels out of the policies,
but makes them available to the evaluator so that principle decisions can be
scored exactly.  The environment has a large candidate pool, discovery and
held-out regions, transient noise, visible-channel surrogate failure, a
generalizable mechanism, a local artifact, and a mechanism that later
reverses.  All randomness is paired and deterministic for seed 0 and seed 1.

The benchmark compares:

* ``source_blind``: expand after three high-surprisal events;
* ``fixed_gated``: repeat + independent verification, but no held-out test;
* ``active_gated``: choose the cheapest distinguishing action, run a held-out
  probe before expansion, and backtrack after later counterevidence;
* ``active_no_backtrack``: the same active gate without route retraction,
  isolating the value of backtracking.

It is deliberately not presented as an official ChEMBL benchmark.  Its role
is to establish causal and algorithmic evidence for the source-aware active
principle-update loop before expensive API runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SEEDS = (0, 1)
BUDGETS = (8, 24, 48)
EVENT_ROUNDS = (5, 6, 7)
BACKTRACK_CHECK_ROUNDS = (16, 28, 40)
ANOMALY_THRESHOLD = 0.85
RESIDUAL_SCALE = 0.15
EVENT_OFFSET = 0.45
REPEAT_TOLERANCE = 0.12
INDEPENDENT_TOLERANCE = 0.12
HELDOUT_GAIN_THRESHOLD = 0.16
HELDOUT_PROBE_COST = 4
POLICIES = ("source_blind", "fixed_gated", "active_gated", "active_no_backtrack")
CONDITIONS = (
    "clean",
    "noise",
    "surrogate_failure",
    "true_mechanism",
    "local_artifact",
    "mechanism_reversal",
)
TRUE_INITIAL_CONDITIONS = {"true_mechanism", "mechanism_reversal"}
NONMECHANISM_CONDITIONS = {"clean", "noise", "surrogate_failure", "local_artifact"}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    x: float
    y: float


@dataclass
class Trial:
    seed: int
    condition: str
    policy: str
    budget: int
    primary_evaluations: int
    total_calls: int
    update_round: int | None
    final_principle: str
    initial_update_correct: bool | None
    final_state_correct: bool | None
    false_expansion: bool
    missed_true_expansion: bool
    backtrack_count: int
    backtrack_rounds: List[int]
    detected_events: int
    detection_misses: int
    allowed_events: int
    false_allows: int
    missed_true_events: int
    verification_cost: int
    heldout_probe_count: int
    heldout_pass_count: int
    heldout_final_gain: float | None
    cumulative_regret: float
    post_update_regret: float
    best_true_reward: float
    event_decisions: List[Dict[str, Any]] = field(default_factory=list)


def build_candidates(seed: int, count: int = 2048) -> List[Candidate]:
    """Generate a paired MBO-style design space with two observable features."""
    rng = random.Random(7919 + seed * 104729)
    return [Candidate(f"x_{i:04d}", rng.random(), rng.random()) for i in range(count)]


def in_trigger_region(c: Candidate) -> bool:
    return 0.70 <= c.x <= 0.82 and 0.70 <= c.y <= 0.82


def in_heldout_region(c: Candidate) -> bool:
    return c.x >= 0.84 and c.y >= 0.84


def in_mechanism_region(c: Candidate) -> bool:
    return c.x >= 0.70 and c.y >= 0.70


def in_surrogate_fault_region(c: Candidate) -> bool:
    return 0.70 <= c.x <= 0.82 and 0.25 <= c.y <= 0.40


def base_value(c: Candidate) -> float:
    # The initial principle is intentionally incomplete but still useful for
    # ordinary MBO ranking.  The hidden mechanism is an interaction term.
    return 0.35 + 0.22 * c.x + 0.08 * (1.0 - c.y) + 0.025 * math.sin(9 * c.x + 3 * c.y)


def mechanism_bump(c: Candidate, condition: str, round_index: int) -> float:
    if condition == "true_mechanism" and in_mechanism_region(c):
        return EVENT_OFFSET
    if condition == "mechanism_reversal" and round_index < 24 and in_mechanism_region(c):
        return EVENT_OFFSET
    if condition == "local_artifact" and in_trigger_region(c):
        return EVENT_OFFSET
    return 0.0


def true_value(c: Candidate, condition: str, round_index: int) -> float:
    return base_value(c) + mechanism_bump(c, condition, round_index)


def principle_prediction(c: Candidate, principle: str) -> float:
    if principle == "P1":
        return base_value(c) + (EVENT_OFFSET if in_mechanism_region(c) else 0.0)
    return base_value(c)


def model_expected(c: Candidate, principle: str) -> float:
    # Anomaly detection is deliberately relative to the currently trusted
    # principle; the hidden source is never used here.
    return principle_prediction(c, principle)


def surprisal(observed: float, expected: float) -> float:
    z = abs(observed - expected) / RESIDUAL_SCALE
    return 1.0 - math.exp(-abs(z))


def event_candidate(candidates: Sequence[Candidate], condition: str, index: int) -> Candidate:
    if condition in {"true_mechanism", "local_artifact", "mechanism_reversal"}:
        pool = [c for c in candidates if in_trigger_region(c)]
    elif condition == "surrogate_failure":
        pool = [c for c in candidates if in_surrogate_fault_region(c)]
    else:
        pool = [c for c in candidates if in_trigger_region(c)]
    if len(pool) < len(EVENT_ROUNDS):
        raise RuntimeError("candidate pool does not contain the required event region")
    return sorted(pool, key=lambda c: c.candidate_id)[index]


def choose_candidate(
    candidates: Sequence[Candidate],
    tested: set[str],
    principle: str,
    rng: random.Random,
) -> Candidate:
    pool = [c for c in candidates if c.candidate_id not in tested]
    if not pool:
        pool = list(candidates)
    # A small exploratory component keeps this a search problem rather than a
    # pure lookup while preserving the principle's downstream effect.
    if rng.random() < 0.12:
        return rng.choice(pool)
    return max(pool, key=lambda c: principle_prediction(c, principle))


def primary_observation(
    c: Candidate,
    condition: str,
    round_index: int,
    event: bool,
    rng: random.Random,
) -> float:
    if event and condition != "clean":
        # All source conditions are matched at the visible channel.  This is
        # the central source-confusion intervention.
        return base_value(c) + EVENT_OFFSET + rng.gauss(0.0, 0.008)
    return true_value(c, condition, round_index) + rng.gauss(0.0, 0.008)


def verify_channel(
    c: Candidate,
    condition: str,
    primary: float,
    round_index: int,
    rng: random.Random,
) -> Dict[str, float]:
    """Return hidden repeat/independent measurements for evaluator scoring."""
    base = base_value(c)
    if condition == "noise":
        return {"repeat": base, "independent": base}
    if condition == "surrogate_failure":
        # The visible channel repeats a faulty value; the independent channel
        # exposes the underlying objective.
        return {"repeat": primary, "independent": base}
    if condition in {"true_mechanism", "mechanism_reversal"}:
        latent = true_value(c, condition, round_index)
        return {"repeat": primary, "independent": latent + rng.gauss(0.0, 0.008)}
    if condition == "local_artifact":
        latent = true_value(c, condition, round_index)
        return {"repeat": primary, "independent": latent + rng.gauss(0.0, 0.008)}
    return {"repeat": base, "independent": base}


def heldout_gain(candidates: Sequence[Candidate], condition: str, round_index: int) -> float:
    heldout = [c for c in candidates if in_heldout_region(c)]
    if not heldout:
        raise RuntimeError("held-out region is empty")
    # Compare the candidate principle's expected gain with the true gain on
    # points never used in the three forced discovery events.
    return statistics.mean(
        true_value(c, condition, round_index) - base_value(c) for c in heldout
    )


def final_correct(condition: str, principle: str) -> bool | None:
    if condition == "clean":
        return principle == "P0"
    if condition in NONMECHANISM_CONDITIONS:
        return principle == "P0"
    if condition == "true_mechanism":
        return principle == "P1"
    # A reversal is deliberately expected to return to the old route.
    if condition == "mechanism_reversal":
        return principle == "P0"
    return None


def initial_correct(condition: str, updated: bool) -> bool | None:
    if condition == "clean":
        return not updated
    if condition in {"noise", "surrogate_failure", "local_artifact"}:
        return not updated
    return updated


def action_for_event(
    *,
    policy: str,
    candidates: Sequence[Candidate],
    event_candidate_value: Candidate,
    condition: str,
    primary: float,
    round_index: int,
    verify_rng: random.Random,
) -> Dict[str, Any]:
    """Run a cost-aware diagnostic policy without exposing source labels."""
    score = surprisal(primary, base_value(event_candidate_value))
    result: Dict[str, Any] = {
        "round": round_index,
        "candidate_id": event_candidate_value.candidate_id,
        "surprisal": score,
        # Evaluator-only instrumentation.  This value is not exposed to the
        # policy and is recorded so that matched visible residuals can be
        # checked across hidden source conditions.
        "visible_residual": primary - base_value(event_candidate_value),
        "action_trace": [],
        "allowed": False,
        "decision": "below_anomaly_threshold",
        "verification_cost": 0,
        "heldout_gain": None,
        "source_posterior": None,
    }
    if score <= ANOMALY_THRESHOLD:
        return result
    if policy == "source_blind":
        result.update({"allowed": True, "decision": "source_blind_allow"})
        return result

    channels = verify_channel(
        event_candidate_value, condition, primary, round_index, verify_rng
    )
    # The posterior is deliberately over observations, not the hidden label.
    posterior = {
        "observation_noise": 0.50,
        "surrogate_local_failure": 0.20,
        "local_or_non_generalizing_mechanism": 0.15,
        "true_mechanism_change": 0.15,
    }
    result["source_posterior"] = dict(posterior)

    def finish() -> Dict[str, Any]:
        # Keep the posterior synchronized with the terminal action.  The
        # benchmark uses this only for evaluator-side calibration metrics.
        result["source_posterior"] = dict(posterior)
        return result

    expected_positive = {
        "repeat": {"observation_noise": 1.0, "surrogate_local_failure": 0.0,
                    "local_or_non_generalizing_mechanism": 0.0, "true_mechanism_change": 0.0},
        "independent": {"observation_noise": 1.0, "surrogate_local_failure": 1.0,
                        "local_or_non_generalizing_mechanism": 0.0, "true_mechanism_change": 0.0},
        "heldout_probe": {"observation_noise": 0.0, "surrogate_local_failure": 0.0,
                          "local_or_non_generalizing_mechanism": 0.0, "true_mechanism_change": 1.0},
    }

    def entropy(p: float) -> float:
        p = max(1e-12, min(1.0 - 1e-12, p))
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    available = ["repeat", "independent"]
    while available:
        if policy == "fixed_gated":
            selected = "repeat" if "repeat" in available else "independent"
            scores = {selected: 0.0}
        else:
            costs = {"repeat": 1.0, "independent": 1.0, "heldout_probe": float(HELDOUT_PROBE_COST)}
            scores = {
                action: entropy(sum(posterior[h] * expected_positive[action][h] for h in posterior)) / costs[action]
                for action in available
            }
            selected = max(scores, key=scores.get)
        result.setdefault("action_selection", []).append({
            "available": list(available), "selected": selected, "scores": scores
        })
        if selected == "repeat":
            repeat_gap = abs(primary - channels["repeat"])
            result["action_trace"].append("repeat")
            result["verification_cost"] += 1
            result["repeat_gap"] = repeat_gap
            available.remove("repeat")
            if repeat_gap > REPEAT_TOLERANCE:
                posterior = {
                    "observation_noise": 0.92,
                    "surrogate_local_failure": 0.03,
                    "local_or_non_generalizing_mechanism": 0.03,
                    "true_mechanism_change": 0.02,
                }
                result["decision"] = "repeat_disagrees_noise"
                return finish()
            posterior = {
                "observation_noise": 0.01,
                "surrogate_local_failure": 0.31,
                "local_or_non_generalizing_mechanism": 0.30,
                "true_mechanism_change": 0.38,
            }
        elif selected == "independent":
            independent_gap = abs(primary - channels["independent"])
            result["action_trace"].append("independent")
            result["verification_cost"] += 1
            result["independent_gap"] = independent_gap
            available.remove("independent")
            if independent_gap > INDEPENDENT_TOLERANCE:
                posterior = {
                    "observation_noise": 0.03,
                    "surrogate_local_failure": 0.92,
                    "local_or_non_generalizing_mechanism": 0.03,
                    "true_mechanism_change": 0.02,
                }
                result["decision"] = "independent_disagrees_surrogate"
                return finish()
            posterior = {
                "observation_noise": 0.01,
                "surrogate_local_failure": 0.02,
                "local_or_non_generalizing_mechanism": 0.48,
                "true_mechanism_change": 0.49,
            }
        else:
            result["action_trace"].append("heldout_probe")
            result["verification_cost"] += HELDOUT_PROBE_COST
            gain = heldout_gain(candidates, condition, round_index)
            result["heldout_gain"] = gain
            if gain >= HELDOUT_GAIN_THRESHOLD:
                posterior = {
                    "observation_noise": 0.02,
                    "surrogate_local_failure": 0.02,
                    "local_or_non_generalizing_mechanism": 0.06,
                    "true_mechanism_change": 0.90,
                }
                result.update({"allowed": True, "decision": "heldout_generalizes"})
            else:
                posterior = {
                    "observation_noise": 0.02,
                    "surrogate_local_failure": 0.02,
                    "local_or_non_generalizing_mechanism": 0.90,
                    "true_mechanism_change": 0.06,
                }
                result["decision"] = "heldout_fails_local_or_spurious"
            return finish()

        if "repeat_gap" in result and "independent_gap" in result:
            if policy == "fixed_gated":
                result.update({"allowed": True, "decision": "fixed_checks_agree"})
                return finish()
            # Held-out is the only remaining active action after both cheap
            # channels agree; the loop still records its value/cost choice.
            available = ["heldout_probe"]
            continue

    if policy == "fixed_gated" and result.get("decision") == "below_anomaly_threshold":
        result.update({"allowed": True, "decision": "fixed_checks_agree"})
    return finish()


def run_trial(seed: int, condition: str, policy: str, budget: int) -> Trial:
    if seed not in SEEDS:
        raise ValueError(f"This frozen experiment only permits seed 0/1, got {seed}")
    candidates = build_candidates(seed)
    by_round = {
        round_index: event_candidate(candidates, condition, index)
        for index, round_index in enumerate(EVENT_ROUNDS)
    }
    rng = random.Random(1009 * seed + 100003 * (CONDITIONS.index(condition) + 1) + len(policy) * 17)
    verify_rng = random.Random(9176 * seed + 113 * CONDITIONS.index(condition) + 29)
    tested: set[str] = set()
    principle = "P0"
    event_decisions: List[Dict[str, Any]] = []
    selected_true_values: List[float] = []
    detected = detection_misses = allowed_events = false_allows = missed_true_events = 0
    verification_cost = heldout_probe_count = heldout_pass_count = 0
    update_round: int | None = None
    backtrack_rounds: List[int] = []
    total_calls = 0
    cumulative_regret = 0.0
    post_update_regret = 0.0
    heldout_final_gain: float | None = None

    for t in range(budget):
        candidate = by_round[t] if t in by_round else choose_candidate(candidates, tested, principle, rng)
        tested.add(candidate.candidate_id)
        event = t in by_round and condition != "clean"
        truth = true_value(candidate, condition, t)
        observed = primary_observation(candidate, condition, t, event, rng)
        selected_true_values.append(truth)
        oracle = max(true_value(c, condition, t) for c in candidates)
        cumulative_regret += oracle - truth
        total_calls += 1

        if event:
            decision = action_for_event(
                policy=policy,
                candidates=candidates,
                event_candidate_value=candidate,
                condition=condition,
                primary=observed,
                round_index=t,
                verify_rng=verify_rng,
            )
            event_decisions.append(decision)
            verification_cost += int(decision["verification_cost"])
            total_calls += int(decision["verification_cost"])
            if decision["surprisal"] > ANOMALY_THRESHOLD:
                detected += 1
                allowed_events += int(decision["allowed"])
                if condition in NONMECHANISM_CONDITIONS:
                    false_allows += int(decision["allowed"])
                if condition in TRUE_INITIAL_CONDITIONS:
                    missed_true_events += int(not decision["allowed"])
                if "heldout_probe" in decision["action_trace"]:
                    heldout_probe_count += 1
                    heldout_pass_count += int(decision["allowed"])
                    heldout_final_gain = decision.get("heldout_gain")
            else:
                detection_misses += 1
                if condition in TRUE_INITIAL_CONDITIONS:
                    missed_true_events += 1

        # Expand only after all three event observations are accepted.  For
        # source-blind/fixed policies this intentionally allows the local
        # artifact; active gating has already tested held-out generalization.
        if t == EVENT_ROUNDS[-1] and len(event_decisions) == len(EVENT_ROUNDS):
            if all(bool(row["allowed"]) for row in event_decisions):
                principle = "P1"
                update_round = t

        if principle == "P1" and t > EVENT_ROUNDS[-1]:
            post_update_regret += oracle - truth

        # Only the active backtracking policy can retract a previously accepted
        # route.  The probe is a counterfactual check on held-out points.
        if (
            policy == "active_gated"
            and principle == "P1"
            and t in BACKTRACK_CHECK_ROUNDS
        ):
            gain = heldout_gain(candidates, condition, t)
            heldout_final_gain = gain
            if gain < HELDOUT_GAIN_THRESHOLD:
                principle = "P0"
                backtrack_rounds.append(t)

    updated = update_round is not None
    initial_update_correct = initial_correct(condition, updated)
    final_state_correct = final_correct(condition, principle)
    false_expansion = condition in NONMECHANISM_CONDITIONS and updated
    missed_true_expansion = condition in TRUE_INITIAL_CONDITIONS and not updated
    return Trial(
        seed=seed,
        condition=condition,
        policy=policy,
        budget=budget,
        primary_evaluations=budget,
        total_calls=total_calls,
        update_round=update_round,
        final_principle=principle,
        initial_update_correct=initial_update_correct,
        final_state_correct=final_state_correct,
        false_expansion=false_expansion,
        missed_true_expansion=missed_true_expansion,
        backtrack_count=len(backtrack_rounds),
        backtrack_rounds=backtrack_rounds,
        detected_events=detected,
        detection_misses=detection_misses,
        allowed_events=allowed_events,
        false_allows=false_allows,
        missed_true_events=missed_true_events,
        verification_cost=verification_cost,
        heldout_probe_count=heldout_probe_count,
        heldout_pass_count=heldout_pass_count,
        heldout_final_gain=heldout_final_gain,
        cumulative_regret=cumulative_regret,
        post_update_regret=post_update_regret,
        best_true_reward=max(selected_true_values),
        event_decisions=event_decisions,
    )


def mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def summarize(rows: Sequence[Trial]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[Trial]]]] = {}
    for row in rows:
        grouped.setdefault(str(row.budget), {}).setdefault(row.condition, {}).setdefault(row.policy, []).append(row)
    output: Dict[str, Any] = {}
    for budget, conditions in grouped.items():
        output[budget] = {}
        for condition, policies in conditions.items():
            output[budget][condition] = {}
            for policy, trials in policies.items():
                output[budget][condition][policy] = {
                    "n": len(trials),
                    "final_state_accuracy": mean(float(t.final_state_correct) for t in trials),
                    "initial_update_accuracy": mean(float(t.initial_update_correct) for t in trials),
                    "false_expansion_rate": mean(float(t.false_expansion) for t in trials),
                    "missed_true_expansion_rate": mean(float(t.missed_true_expansion) for t in trials),
                    "mean_backtrack_count": mean(t.backtrack_count for t in trials),
                    "mean_detected_events": mean(t.detected_events for t in trials),
                    "mean_detection_misses": mean(t.detection_misses for t in trials),
                    "mean_false_allows": mean(t.false_allows for t in trials),
                    "mean_missed_true_events": mean(t.missed_true_events for t in trials),
                    "mean_verification_cost": mean(t.verification_cost for t in trials),
                    "mean_total_calls": mean(t.total_calls for t in trials),
                    "mean_heldout_probe_count": mean(t.heldout_probe_count for t in trials),
                    "mean_heldout_pass_count": mean(t.heldout_pass_count for t in trials),
                    "mean_heldout_final_gain": mean(t.heldout_final_gain for t in trials),
                    "mean_cumulative_regret": mean(t.cumulative_regret for t in trials),
                    "mean_post_update_regret": mean(t.post_update_regret for t in trials),
                    "mean_best_true_reward": mean(t.best_true_reward for t in trials),
                }
    return output


def checkpoint_rows(rows: Sequence[Trial]) -> List[Dict[str, Any]]:
    """Re-run prefixes to expose 8/24/48 learning curves from one protocol."""
    result: List[Dict[str, Any]] = []
    for row in rows:
        # Prefixes are independent deterministic runs with the same frozen
        # seed/configuration; this avoids mixing a separate 24-budget rerun.
        prefix = run_trial(row.seed, row.condition, row.policy, row.budget)
        result.append({
            "seed": row.seed,
            "condition": row.condition,
            "policy": row.policy,
            "budget": row.budget,
            "primary_evaluations": prefix.primary_evaluations,
            "total_calls": prefix.total_calls,
            "final_principle": prefix.final_principle,
            "final_state_correct": prefix.final_state_correct,
            "false_expansion": prefix.false_expansion,
            "backtrack_count": prefix.backtrack_count,
            "verification_cost": prefix.verification_cost,
            "cumulative_regret": prefix.cumulative_regret,
            "post_update_regret": prefix.post_update_regret,
        })
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def proof_checks(rows: Sequence[Trial]) -> Dict[str, Any]:
    """Machine-check the benchmark's intended decision properties."""
    checks: Dict[str, Any] = {}
    checks["seed_scope"] = sorted({r.seed for r in rows}) == [0, 1]
    checks["all_budgets_present"] = sorted({r.budget for r in rows}) == list(BUDGETS)
    def rows_for(policy: str, condition: str, minimum_budget: int = 0) -> List[Trial]:
        return [
            r for r in rows
            if r.policy == policy and r.condition == condition and r.budget >= minimum_budget
        ]

    checks["source_blind_false_expands_spurious"] = all(
        all(r.false_expansion for r in rows_for("source_blind", c))
        for c in ("noise", "surrogate_failure", "local_artifact")
    )
    checks["active_rejects_spurious"] = all(
        all(not r.false_expansion for r in rows_for("active_gated", c))
        for c in ("noise", "surrogate_failure", "local_artifact")
    )
    checks["active_accepts_true_mechanism"] = all(
        r.final_state_correct for r in rows_for("active_gated", "true_mechanism")
    )
    checks["active_backtracks_reversal"] = all(
        r.backtrack_count > 0 and r.final_state_correct
        for r in rows_for("active_gated", "mechanism_reversal", minimum_budget=48)
    )
    checks["no_backtrack_ablation_fails_reversal"] = all(
        not r.final_state_correct
        for r in rows_for("active_no_backtrack", "mechanism_reversal")
    )
    checks["all_checks_pass"] = all(checks.values())
    return checks


def write_plot_data(output_dir: Path, checkpoint: Sequence[Mapping[str, Any]]) -> None:
    # Plotting is kept in a separate optional module; this JSON makes the
    # numerical artifact available even on machines without matplotlib.
    (output_dir / "checkpoint_metrics.json").write_text(
        json.dumps(list(checkpoint), indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    if sorted(args.seeds) != [0, 1]:
        raise SystemExit("The frozen main experiment must run exactly seed0 and seed1.")

    rows: List[Trial] = []
    for budget in BUDGETS:
        for seed in args.seeds:
            for condition in CONDITIONS:
                for policy in POLICIES:
                    rows.append(run_trial(seed, condition, policy, budget))
    checkpoint = checkpoint_rows(rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    serialised = [asdict(row) for row in rows]
    (output_dir / "trial_results.json").write_text(json.dumps(serialised, indent=2), encoding="utf-8")
    write_csv(output_dir / "trial_results.csv", serialised)
    write_csv(output_dir / "checkpoint_metrics.csv", checkpoint)
    write_plot_data(output_dir, checkpoint)
    summary = {
        "description": "Controlled hidden-law MBO benchmark for reliable active principle discovery; not an official PiEvo leaderboard.",
        "seeds": list(args.seeds),
        "budgets": list(BUDGETS),
        "event_rounds": list(EVENT_ROUNDS),
        "backtrack_check_rounds": list(BACKTRACK_CHECK_ROUNDS),
        "conditions": list(CONDITIONS),
        "policies": list(POLICIES),
        "candidate_count": 2048,
        "heldout_gain_threshold": HELDOUT_GAIN_THRESHOLD,
        "summary": summarize(rows),
        "proof_checks": proof_checks(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "protocol": "RADP-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": summary | {"output_dir": str(output_dir.resolve())},
        "claim_boundary": [
            "This controlled benchmark establishes source attribution and active principle-update behavior under known hidden laws.",
            "It does not establish official ChEMBL performance or physical chemical discovery.",
            "The LLM integration results remain a separate ecological-validity layer.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "proof_checks": summary["proof_checks"]}, indent=2))
    if not summary["proof_checks"]["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
