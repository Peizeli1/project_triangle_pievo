from __future__ import annotations

import json
from pathlib import Path

from pievo.group.evolveflow import PiEvo, RADP_WIDE_SEED_STREAM
from pievo.word.instruction import get_hypothesis_guidance_prompt


def _strategy(tmp_path: Path) -> PiEvo:
    strategy = object.__new__(PiEvo)
    strategy.task = "Molecular Bio-activity Optimization (MBO): maximize pChEMBL"
    strategy.output_dir = str(tmp_path)
    strategy.history = []
    strategy.submissions = [
        {
            "source_agent": "hypothesis",
            "json_data": {
                "HYPOTHESIS_SUBMISSION_A": {
                    "candidate": "c1ccccc1",
                    "rational": "local",
                },
                "HYPOTHESIS_SUBMISSION_B": {
                    "candidate": "C1CCCCCCC1",
                    "rational": "cross scaffold",
                },
                "HYPOTHESIS_SUBMISSION_C": {
                    "candidate": "C1C2CC3CC1CC(C2)C3",
                    "rational": "orthogonal",
                },
            },
        }
    ]
    strategy._radp_portfolio_selected_roles = []
    strategy.round_by_PHE_order_before_P = 0
    return strategy


def test_portfolio_strategy_consumes_fixed_roles_and_audits(tmp_path, monkeypatch) -> None:
    strategy = _strategy(tmp_path)
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")

    candidates = ["c1ccccc1", "C1CCCCCCC1", "C1C2CC3CC1CC(C2)C3"]
    selected = [
        strategy._select_hypothesis_portfolio(candidates, reward_guided=False)
        for _ in range(3)
    ]

    assert selected == candidates
    assert strategy._radp_portfolio_selected_roles == [
        "local_exploitation",
        "cross_scaffold",
        "orthogonal_exploration",
    ]

    rows = [
        json.loads(line)
        for line in (tmp_path / "radp_candidate_portfolio_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["target_role"] for row in rows] == strategy._radp_portfolio_selected_roles
    assert all(row["target_role_available"] for row in rows)
    assert all(row["provisional_principles_used_for_selection"] is False for row in rows)
    assert all(row["source_gate_values_used_for_selection"] is False for row in rows)
    assert all(row["verifier_values_used_for_selection"] is False for row in rows)


def test_portfolio_strategy_fails_closed_to_deterministic_fallback(tmp_path, monkeypatch) -> None:
    strategy = _strategy(tmp_path)
    strategy.submissions[0]["json_data"].pop("HYPOTHESIS_SUBMISSION_B")
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")

    selected = strategy._select_hypothesis_portfolio(
        ["c1ccccc1", "C1C2CC3CC1CC(C2)C3"], reward_guided=False
    )

    assert selected in {"c1ccccc1", "C1C2CC3CC1CC(C2)C3"}
    row = json.loads(
        (tmp_path / "radp_candidate_portfolio_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["target_role"] == "local_exploitation"
    assert row["target_role_available"] is True


def test_conflicting_role_slots_become_unlabeled(tmp_path, monkeypatch) -> None:
    strategy = _strategy(tmp_path)
    strategy.submissions[0]["json_data"]["HYPOTHESIS_SUBMISSION_B"]["candidate"] = "c1ccccc1"
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")

    selected = strategy._select_hypothesis_portfolio(
        ["c1ccccc1", "C1C2CC3CC1CC(C2)C3"], reward_guided=False
    )

    assert selected in {"c1ccccc1", "C1C2CC3CC1CC(C2)C3"}
    row = json.loads(
        (tmp_path / "radp_candidate_portfolio_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["target_role_available"] is False
    assert row["role_counts"]["unlabeled"] == 1


def test_portfolio_strategy_skips_to_next_available_role(tmp_path, monkeypatch) -> None:
    strategy = _strategy(tmp_path)
    strategy.submissions[0]["json_data"].pop("HYPOTHESIS_SUBMISSION_A")
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")

    selected = strategy._select_hypothesis_portfolio(
        ["C1CCCCCCC1", "C1C2CC3CC1CC(C2)C3"], reward_guided=False
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "radp_candidate_portfolio_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert selected == "C1CCCCCCC1"
    assert rows[0]["schema_version"] == "radp_candidate_portfolio_audit.v2"
    assert rows[0]["target_role"] == "local_exploitation"
    assert rows[0]["target_role_available"] is False
    assert rows[0]["selected_role"] == "cross_scaffold"
    assert rows[0]["fallback_used"] is True
    assert rows[0]["fallback_from_role"] == "local_exploitation"
    assert rows[0]["fallback_to_role"] == "cross_scaffold"
    assert strategy._radp_portfolio_cursor == 2


def test_first_cycle_controller_prioritizes_seed_bank(tmp_path, monkeypatch) -> None:
    strategy = _strategy(tmp_path)
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")
    monkeypatch.setenv("PIEVO_ACCEPTED_PRINCIPLE_ONLY", "1")
    monkeypatch.setenv("PIEVO_TASK", "MBO")

    strategy._inject_radp_seed_bank(strategy.submissions, "hypothesis")
    candidates = [
        "c1ccccc1",
        "C1CCCCCCC1",
        "C1C2CC3CC1CC(C2)C3",
        "c1ccc2cc3cc4ccccc4cc3cc2c1",
        "O=C1CCCCCCCCCCC1",
        "C1CCC2(CC1)CCCCC2",
    ]
    selected = strategy._select_hypothesis_portfolio(candidates, reward_guided=False)
    row = json.loads(
        (tmp_path / "radp_candidate_portfolio_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )

    assert selected == "c1ccc2cc3cc4ccccc4cc3cc2c1"
    assert row["seed_bank_prioritized"] is True
    assert row["selected_role"] == "local_exploitation"


def test_wide_seed_stream_is_outcome_blind_and_consumed_in_order(tmp_path, monkeypatch) -> None:
    strategy = _strategy(tmp_path)
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")
    monkeypatch.setenv("PIEVO_ACCEPTED_PRINCIPLE_ONLY", "1")
    monkeypatch.setenv("PIEVO_TASK", "MBO")
    monkeypatch.setenv("PIEVO_RADP_WIDE_SEED_BANK", "1")

    strategy._inject_radp_seed_bank(strategy.submissions, "hypothesis")
    stream = [candidate for _, candidate in RADP_WIDE_SEED_STREAM]
    selected = []
    remaining = list(stream)
    for _ in stream:
        value = strategy._select_hypothesis_portfolio(remaining, reward_guided=False)
        selected.append(value)
        remaining.remove(value)

    assert selected == stream
    rows = [
        json.loads(line)
        for line in (tmp_path / "radp_candidate_portfolio_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line).get("schema_version") == "radp_candidate_portfolio_audit.v2"
    ]
    assert len(rows) == len(stream)
    assert all(row["seed_stream_enabled"] is True for row in rows)
    assert all(row["seed_stream_id"] == "wide_pool_seed1_limit5000_structural_portfolio_v1" for row in rows)
    assert [row["seed_stream_index"] for row in rows] == list(range(len(stream)))
    assert all(row["source_gate_values_used_for_selection"] is False for row in rows)
    assert all(row["verifier_values_used_for_selection"] is False for row in rows)


def test_provisional_principles_are_recorded_as_proposal_only(tmp_path, monkeypatch) -> None:
    strategy = _strategy(tmp_path)
    strategy.principles = {"p0": "temporary", "p1": "accepted"}
    strategy.principle_beliefs = {"p0": 0.6, "p1": 0.4}
    strategy._radp_runtime = type(
        "Runtime",
        (),
        {"principle_status": lambda self, principle_id: "accepted" if principle_id == "p1" else "provisional"},
    )()
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_ACCEPTED_PRINCIPLE_ONLY", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")

    guidance_principles, _ = strategy._hypothesis_guidance_principle_state()
    strategy._record_principle_guidance_boundary(guidance_principles)

    row = json.loads(
        (tmp_path / "radp_candidate_portfolio_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["schema_version"] == "radp_principle_guidance_boundary.v2"
    assert row["guidance_principle_ids"] == ["p1"]
    assert row["excluded_internal_principle_ids"] == ["p0"]
    assert row["all_guidance_principles_are_accepted"] is True
    assert row["proposal_only"] is True
    assert row["source_gate_payload_exposed"] is False
    assert row["verifier_payload_exposed"] is False


def test_portfolio_prompt_declares_fixed_roles(monkeypatch) -> None:
    monkeypatch.setenv("PIEVO_ACCEPTED_PRINCIPLE_ONLY", "1")
    monkeypatch.setenv("PIEVO_NEUTRAL_REWARD_GUIDED", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "portfolio_v2")
    guidance_type, guidance = get_hypothesis_guidance_prompt(
        task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
        top_principle_text="",
        second_best_principle_text="",
        principle_beliefs={},
        num_principles=0,
    )

    assert guidance_type == "NEUTRAL_DIVERSITY"
    assert "A is one local" in guidance
    assert "B is one high-prior cross-scaffold" in guidance
    assert "C is one orthogonal exploration" in guidance
    assert "verifier or source-gate details" in guidance


def test_first_neutral_cycle_declares_outcome_blind_seed_bank(monkeypatch) -> None:
    monkeypatch.setenv("PIEVO_ACCEPTED_PRINCIPLE_ONLY", "1")
    monkeypatch.setenv("PIEVO_NEUTRAL_REWARD_GUIDED", "1")
    guidance_type, guidance = get_hypothesis_guidance_prompt(
        task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
        top_principle_text="",
        second_best_principle_text="",
        principle_beliefs={},
        num_principles=0,
        candidates_done=[],
    )

    assert guidance_type == "NEUTRAL_DIVERSITY"
    assert "PRE-REGISTERED NEUTRAL SEED PORTFOLIO" in guidance
    assert "c1ccc2cc3cc4ccccc4cc3cc2c1" in guidance
    assert "O=C1CCCCCCCCCCC1" in guidance
    assert "C1CCC2(CC1)CCCCC2" in guidance
    assert "outcome-blind structural prior" in guidance
