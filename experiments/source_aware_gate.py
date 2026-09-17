#!/usr/bin/env python3
"""Minimal, auditable source-attribution gate for direction 1.

The gate deliberately has no learned weights and no access to the hidden
condition label.  It consumes three numeric observations:

* ``primary``: the observation that produced a high surprisal;
* ``repeat``: a repeat through the visible observation channel;
* ``independent``: an independent/high-fidelity verification channel.

The decision rule is intentionally small enough to audit:

1. a primary--repeat mismatch is attributed to transient observation noise;
2. a stable repeat that disagrees with the independent channel is attributed
   to a local surrogate failure;
3. agreement across the two channels is treated as evidence for a persistent
   mechanism change and permits a principle update.

The class never receives ``source_label``.  Labels are accepted only by the
evaluation code when computing false/missed-update metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GateDecision:
    source_decision: str
    update_allowed: bool
    reason: str
    repeat_gap: float
    independent_gap: float
    verification_cost: int = 2
    source_posterior: Optional[Dict[str, float]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SourceAwareGate:
    """A deterministic two-check source attribution and update gate."""

    def __init__(self, tolerance: float = 0.25):
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.tolerance = float(tolerance)

    def decide(
        self,
        *,
        primary: float,
        repeat: float,
        independent: float,
        surprisal: Optional[float] = None,
        threshold: float = 0.85,
    ) -> GateDecision:
        """Return a source decision without using an injected label.

        A gate is only meaningful for a high-surprisal observation.  Ordinary
        observations are explicitly rejected for principle mutation so that a
        verification query cannot itself become an update trigger.
        """
        repeat_gap = abs(float(primary) - float(repeat))
        independent_gap = abs(float(primary) - float(independent))
        if surprisal is not None and float(surprisal) <= float(threshold):
            return GateDecision(
                source_decision="no_anomaly",
                update_allowed=False,
                reason="surprisal_below_threshold",
                repeat_gap=repeat_gap,
                independent_gap=independent_gap,
                verification_cost=1,
                source_posterior={
                    "observation_noise": 0.90,
                    "surrogate_local_failure": 0.08,
                    "true_mechanism_change": 0.02,
                },
            )

        if repeat_gap > self.tolerance:
            return GateDecision(
                source_decision="observation_noise",
                update_allowed=False,
                reason="repeat_channel_disagrees",
                repeat_gap=repeat_gap,
                independent_gap=independent_gap,
                source_posterior={
                    "observation_noise": 0.05,
                    "surrogate_local_failure": 0.90,
                    "true_mechanism_change": 0.05,
                },
            )

        if independent_gap > self.tolerance:
            return GateDecision(
                source_decision="surrogate_local_failure",
                update_allowed=False,
                reason="persistent_visible_value_disagrees_with_independent_check",
                repeat_gap=repeat_gap,
                independent_gap=independent_gap,
            )

        return GateDecision(
            source_decision="true_mechanism_change",
            update_allowed=True,
            reason="repeat_and_independent_check_agree",
            repeat_gap=repeat_gap,
            independent_gap=independent_gap,
            source_posterior={
                "observation_noise": 0.03,
                "surrogate_local_failure": 0.07,
                "true_mechanism_change": 0.90,
            },
        )


def verification_values(
    *,
    base_surrogate: float,
    reference_value: float,
    observed_outcome: float,
    condition: str,
    event: bool,
    offset: float = 2.0,
) -> tuple[float, float]:
    """Construct controlled repeat/independent observations for replay.

    This function is an evaluator-side replay helper.  The condition is used
    to construct the hidden verification channel and must not be exposed to an
    agent.  A real deployment would replace it with two actual tool calls.
    """
    if not event:
        return float(base_surrogate), float(reference_value)
    if condition == "noise":
        # The injected one-shot spike disappears on a repeat.
        return float(base_surrogate), float(reference_value)
    if condition == "surrogate_failure":
        # A local surrogate fault persists in the visible channel, but not in
        # an independent high-fidelity/reference channel.
        return float(observed_outcome), float(reference_value)
    if condition == "true_mechanism":
        # The independent channel confirms the persistent mechanism shift.
        return float(observed_outcome), float(reference_value) + float(offset)
    return float(base_surrogate), float(reference_value)
