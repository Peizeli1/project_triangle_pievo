"""Offline tests for the exploratory controlled-MBO gate policies."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import controlled_mbo_policy_controls as controls  # noqa: E402


def test_adaptive_gate_rejects_source_only_spike() -> None:
    allowed, cost, reason = controls.gate_event(
        "adaptive_gate",
        primary=0.94,
        repeat=0.25,
        independent=0.25,
        expected=0.25,
    )
    assert allowed is False
    assert cost == 2
    assert reason == "adaptive_verifier_disagreement_or_weak_gain"


def test_adaptive_gate_recovers_primary_spike_with_positive_verifier_consensus() -> None:
    allowed, cost, reason = controls.gate_event(
        "adaptive_gate",
        primary=1.08,
        repeat=0.90,
        independent=0.91,
        expected=0.25,
    )
    assert allowed is True
    assert cost == 2
    assert reason == "adaptive_primary_spike_verified_consensus"


def test_adaptive_gate_accepts_weak_effect_when_all_channels_agree() -> None:
    allowed, cost, reason = controls.gate_event(
        "adaptive_gate",
        primary=0.35,
        repeat=0.35,
        independent=0.35,
        expected=0.27,
    )
    assert allowed is True
    assert cost == 2
    assert reason == "adaptive_all_channels_agree"


def test_adaptive_candidate_stress_recovery_is_separate_from_frozen_gate() -> None:
    result = controls.run_trial(0, "mixed_noise_mechanism", "adaptive_gate", "fixed_total")
    assert result.update_correct is True
    assert result.policy == "adaptive_gate"
    assert result.verification_cost > 0

