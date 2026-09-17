from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RADP_REPO = ROOT / "official_sources/PiEvo_radp_qwen235b"
if str(RADP_REPO) not in sys.path:
    sys.path.insert(0, str(RADP_REPO))

from pievo.radp.afe import (  # noqa: E402
    ROLE_ORDER,
    generate_afe_portfolio,
    outcome_blind_initial_priors,
    select_afe_candidate,
)
from pievo.group.evolveflow import PiEvo  # noqa: E402
from pievo.tools.candidate_validation import validate_mbo  # noqa: E402


def test_afe_mbo_portfolio_is_deterministic_valid_and_three_slot() -> None:
    history = [
        ("c1ccccc1", 1.0),
        ("Cc1ccccc1", 2.0),
        ("C1CCCCCCC1", 1.5),
    ]
    first = generate_afe_portfolio(task="MBO", history=history, seed=4, budget=12)
    second = generate_afe_portfolio(task="MBO", history=history, seed=4, budget=12)

    assert first == second
    assert len(first) == 12
    candidates = [row["candidate"] for row in first]
    assert len(candidates) == len(set(candidates))
    assert all(validate_mbo(candidate).canonical == candidate for candidate in candidates)
    assert {row["role"] for row in first} == set(ROLE_ORDER)
    assert all(row["labels_used"] is False for row in first)
    assert all(
        row["operation"]
        in {
            "mutation",
            "scaffold_expansion",
            "crossover",
            "model_proposal",
            "structural_pool_proposal",
            "outcome_blind_initial_prior",
        }
        for row in first
    )


def test_afe_v8_keeps_fixed_initial_priors_outcome_blind() -> None:
    rows = generate_afe_portfolio(task="MBO", history=[], seed=0, budget=3)
    assert tuple((row["role"], row["candidate"]) for row in rows) == outcome_blind_initial_priors("MBO")
    assert all(row["initial_prior"] is True for row in rows)
    assert all(row["labels_used"] is False for row in rows)


def test_afe_non_mbo_candidates_use_the_task_validator() -> None:
    nho = "fiber_radius=30,helix_radius=45,n_turns=5,pitch=100,wavelength=600,x_y_z=1,direction=0"
    spo = "La2CuO4"
    for task, candidate in (("NHO", nho), ("SPO", spo)):
        rows = generate_afe_portfolio(
            task=task,
            history=[],
            candidate_pool=[candidate],
            seed=0,
            budget=1,
        )
        assert rows[0]["candidate"] == candidate
        assert rows[0]["labels_used"] is False


def test_afe_non_mbo_stream_replenishes_after_history_is_exhausted() -> None:
    first = generate_afe_portfolio(task="SPO", history=[], seed=0, budget=18)
    first_candidates = [row["candidate"] for row in first]
    assert len(first_candidates) == 18
    assert len(set(first_candidates)) == 18
    assert all(row["source"] == "afe_generated_structure" for row in first)

    history = [(candidate, float(index)) for index, candidate in enumerate(first_candidates)]
    second = generate_afe_portfolio(task="SPO", history=history, seed=0, budget=18)
    second_candidates = [row["candidate"] for row in second]
    assert len(second_candidates) == 18
    assert len(set(second_candidates)) == 18
    assert set(first_candidates).isdisjoint(second_candidates)


def test_afe_selection_audit_has_no_verifier_or_hidden_label_inputs() -> None:
    rows = generate_afe_portfolio(
        task="MBO",
        history=[("c1ccccc1", 1.0)],
        seed=0,
        budget=6,
    )
    candidates = [row["candidate"] for row in rows]
    metadata = {row["candidate"]: row for row in rows}
    selected, audit = select_afe_candidate(
        candidates,
        task="MBO",
        history=[("c1ccccc1", 1.0)],
        cursor=0,
        metadata=metadata,
    )

    assert selected in candidates
    assert audit["schema_version"] == "radp_afe_acquisition.v8"
    assert audit["clean_primary_history_only"] is True
    assert audit["verifier_values_used"] is False
    assert audit["source_gate_values_used"] is False
    assert audit["hidden_labels_used"] is False
    assert "verifier" not in json.dumps(audit["selected_candidate"]).lower()


def test_afe_v8_forces_the_target_initial_prior_during_cold_start() -> None:
    candidates = [
        "c1ccc2cc3cc4ccccc4cc3cc2c1",
        "NC(=O)c1ccc2cc3ccccc3cc2c1",
    ]
    metadata = {
        candidates[0]: {
            "role": ROLE_ORDER[0],
            "initial_prior": True,
            "prior_source": "fixed_outcome_blind_mbo_seed_bank",
        },
        candidates[1]: {"role": ROLE_ORDER[0], "initial_prior": False},
    }
    selected, audit = select_afe_candidate(
        candidates,
        task="MBO",
        history=[],
        cursor=0,
        metadata=metadata,
    )

    assert selected == candidates[0]
    assert audit["initial_prior_selected"] is True
    assert audit["bootstrap_prior_forced"] is True
    assert audit["strategy_version"] == "afe_v8"


def test_afe_v8_deduplicates_pool_proposals_that_match_initial_priors() -> None:
    rows = generate_afe_portfolio(
        task="MBO",
        history=[],
        candidate_pool=[
            "c1ccc2cc3cc4ccccc4cc3cc2c1",
            "c1ccc2cc3cc4ccccc4cc3cc2c1",
            "CC",
        ],
        seed=0,
        budget=6,
    )
    candidates = [row["candidate"] for row in rows]
    assert len(candidates) == len(set(candidates))
    assert candidates.count("c1ccc2cc3cc4ccccc4cc3cc2c1") == 1


def test_afe_metadata_preserves_initial_prior_when_stream_repeats_candidate() -> None:
    strategy = object.__new__(PiEvo)
    strategy._radp_afe_metadata = {}

    prior = {
        "candidate": "c1ccccc1",
        "initial_prior": True,
        "prior_source": "fixed_outcome_blind_mbo_seed_bank",
        "operation": "outcome_blind_initial_prior",
    }
    duplicate = {
        "candidate": "c1ccccc1",
        "initial_prior": False,
        "operation": "model_proposal",
    }
    key = strategy._candidate_key(prior["candidate"])
    strategy._radp_afe_metadata[key] = dict(prior)
    previous = strategy._radp_afe_metadata[key]
    merged = dict(duplicate)
    merged.update(
        {
            "initial_prior": True,
            "prior_source": previous["prior_source"],
            "operation": previous["operation"],
        }
    )
    strategy._radp_afe_metadata[key] = merged

    assert strategy._radp_afe_metadata[key]["initial_prior"] is True
    assert strategy._radp_afe_metadata[key]["operation"] == "outcome_blind_initial_prior"


def test_afe_controller_rejects_invalid_model_proposals_before_evaluator(
    tmp_path: Path,
) -> None:
    strategy = object.__new__(PiEvo)
    strategy.task = "Molecular Bio-activity Optimization (MBO)"
    strategy.history = []
    strategy.output_dir = str(tmp_path)

    accepted = strategy._filter_afe_model_candidates(
        ["invalid [", "C=O", "O=C"]
    )

    assert accepted == [validate_mbo("C=O").canonical]
    rows = [
        json.loads(line)
        for line in (tmp_path / "radp_afe_proposal_audit.jsonl").read_text().splitlines()
    ]
    assert [row["disposition"] for row in rows] == [
        "afe_invalid_proposal",
        "afe_canonical_duplicate",
    ]
    assert all(row["entered_evaluator"] is False for row in rows)


def test_common_generator_control_is_forced_before_radp_afe() -> None:
    strategy = object.__new__(PiEvo)
    strategy.task = "Molecular Bio-activity Optimization (MBO)"
    strategy.history = []
    strategy.output_dir = ""
    strategy._common_generator_cursor = 0

    from experiments.common_candidate_generator import generate

    import os

    previous = os.environ.get("PIEVO_COMMON_GENERATOR")
    previous_gate = os.environ.get("PIEVO_SOURCE_GATE")
    previous_strategy = os.environ.get("PIEVO_RADP_SEARCH_STRATEGY")
    try:
        os.environ["PIEVO_COMMON_GENERATOR"] = "1"
        os.environ["PIEVO_SOURCE_GATE"] = "1"
        os.environ["PIEVO_RADP_SEARCH_STRATEGY"] = "afe_v8"
        candidates = strategy._common_generator_candidates()
        assert candidates[0] == generate(task="MBO", seed=0, slot=0, count=3)[0]
        assert strategy._radp_afe_strategy_enabled() is False
    finally:
        for name, value in (
            ("PIEVO_COMMON_GENERATOR", previous),
            ("PIEVO_SOURCE_GATE", previous_gate),
            ("PIEVO_RADP_SEARCH_STRATEGY", previous_strategy),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
