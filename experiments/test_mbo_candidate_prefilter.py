import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from pievo.agents.experiment import _prefilter_mbo_candidates, _prefilter_task_candidates
from pievo.group.evolveflow import PiEvo
from pievo.tools.candidate_validation import CandidateValidationError, validate_mbo
from pievo.word.instruction import (
    MBO_REJECTION_FEEDBACK_LIMIT,
    MBO_VALIDATED_SCAFFOLDS,
    get_hypothesis_guidance_prompt,
    load_mbo_candidate_rejections,
)
from experiments.write_run_manifest import audit_summary
from experiments.outcome_blind_candidate_recovery import (
    CONTRACT_ID,
    recovery_stream,
)
from experiments.mbo_applicability_domain import assess_mbo_applicability


ROOT = Path(__file__).resolve().parents[1]


def test_mbo_prefilter_rejects_invalid_and_canonical_duplicate_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PIEVO_TASK", "MBO")
    monkeypatch.setenv("PIEVO_PREFILTER_INVALID_CANDIDATES", "1")
    monkeypatch.setenv("PIEVO_MAX_PREFILTER_REJECTIONS", "12")
    monkeypatch.setenv("WORKING_RESULT_DIR", str(tmp_path))

    candidates = ["C1=CC=CC=C1", "not a smiles [", "CCO"]
    accepted = _prefilter_mbo_candidates(candidates, ["c1ccccc1"])
    assert accepted == ["CCO"]

    path = tmp_path / "candidate_prefilter_audit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["disposition"] for row in rows] == [
        "canonical_duplicate",
        "invalid",
    ]

    accepted_again = _prefilter_mbo_candidates(candidates, ["c1ccccc1"])
    assert accepted_again == ["CCO"]
    assert len(path.read_text().splitlines()) == 2


def test_mbo_prefilter_fails_closed_after_bounded_all_invalid_pool(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PIEVO_TASK", "MBO")
    monkeypatch.setenv("PIEVO_PREFILTER_INVALID_CANDIDATES", "1")
    monkeypatch.setenv("PIEVO_MAX_PREFILTER_REJECTIONS", "1")
    monkeypatch.setenv("WORKING_RESULT_DIR", str(tmp_path))
    monkeypatch.setenv("PIEVO_STOP_REQUESTED", "0")

    assert _prefilter_mbo_candidates(["invalid ["], []) == []
    assert __import__("os").environ["PIEVO_STOP_REQUESTED"] == "1"


def test_tmc_prefilter_uses_released_validator_without_an_evaluator_call(
    tmp_path: Path, monkeypatch
) -> None:
    from pievo.tools.candidate_validation import (
        CandidateValidationError,
        ValidationResult,
    )
    import pievo.tools.candidate_validation as validation

    calls: list[tuple[str, str, Path | None]] = []

    def fake_validate_candidate(
        task: str, candidate: str, *, surrogate_root: Path | None = None
    ) -> ValidationResult:
        calls.append((task, candidate, surrogate_root))
        if candidate == "Pd_invalid":
            raise CandidateValidationError("TMC total ligand charge must be -2")
        return ValidationResult(task, candidate, candidate.lower(), candidate)

    monkeypatch.setattr(validation, "validate_candidate", fake_validate_candidate)
    monkeypatch.setenv("PIEVO_TASK", "TMC")
    monkeypatch.setenv("PIEVO_PREFILTER_INVALID_CANDIDATES", "1")
    monkeypatch.setenv("PIEVO_SURROGATE_ROOT", str(tmp_path / "released_envs"))
    monkeypatch.setenv("WORKING_RESULT_DIR", str(tmp_path))

    accepted = _prefilter_task_candidates(
        ["Pd_valid", "Pd_invalid", "Pd_DONE"], ["Pd_done"]
    )

    assert accepted == ["Pd_valid"]
    assert calls and all(call[0] == "TMC" for call in calls)
    assert all(call[2] == tmp_path / "released_envs" for call in calls)
    rows = [
        json.loads(line)
        for line in (tmp_path / "candidate_prefilter_audit.jsonl").read_text().splitlines()
    ]
    assert [(row["task"], row["disposition"]) for row in rows] == [
        ("TMC", "invalid"),
        ("TMC", "canonical_duplicate"),
    ]


def test_spo_prefilter_rejects_out_of_system_formula_before_evaluator(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PIEVO_TASK", "SPO")
    monkeypatch.setenv("PIEVO_PREFILTER_INVALID_CANDIDATES", "1")
    monkeypatch.setenv("PIEVO_MAX_PREFILTER_REJECTIONS", "12")
    monkeypatch.setenv("WORKING_RESULT_DIR", str(tmp_path))

    accepted = _prefilter_task_candidates(
        ["Tl1Ba2Ca0Cu3O7", "La1Ba2Cu3O7", "La1Ba2Cu3O7"], []
    )

    assert accepted == ["La1Ba2Cu3O7"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "candidate_prefilter_audit.jsonl").read_text().splitlines()
    ]
    assert [row["disposition"] for row in rows] == ["invalid", "canonical_duplicate"]
    assert all(row["task"] == "SPO" for row in rows)


def test_spo_afe_empty_model_batch_does_not_trigger_shared_stop(
    tmp_path: Path, monkeypatch
) -> None:
    import os
    import pievo.agents.experiment as experiment_module

    monkeypatch.setattr(experiment_module, "_CONSECUTIVE_EMPTY_PREFILTER_POOLS", 0)
    monkeypatch.setenv("PIEVO_TASK", "SPO")
    monkeypatch.setenv("PIEVO_SOURCE_GATE", "1")
    monkeypatch.setenv("PIEVO_RADP_SEARCH_STRATEGY", "afe_spo_v2")
    monkeypatch.setenv("PIEVO_PREFILTER_INVALID_CANDIDATES", "1")
    monkeypatch.setenv("PIEVO_MAX_PREFILTER_REJECTIONS", "1")
    monkeypatch.setenv("PIEVO_MAX_PREFILTER_EMPTY_POOLS", "1")
    monkeypatch.setenv("PIEVO_STOP_REQUESTED", "0")
    monkeypatch.setenv("WORKING_RESULT_DIR", str(tmp_path))

    assert _prefilter_task_candidates([], []) == []
    assert os.environ["PIEVO_STOP_REQUESTED"] == "0"
    assert not (tmp_path / "candidate_prefilter_audit.jsonl").exists()


def test_retained_mbo_development_candidate_cannot_be_recounted(
    monkeypatch,
) -> None:
    retained = "C1CCCCCCC2C1CCCCCCC3C2CCCCCCC3"
    monkeypatch.setenv("PIEVO_MBO_FORBIDDEN_CANDIDATES", retained)
    with pytest.raises(CandidateValidationError, match="retained development"):
        validate_mbo(retained)

    monkeypatch.delenv("PIEVO_MBO_FORBIDDEN_CANDIDATES", raising=False)
    assert validate_mbo(retained).canonical


@pytest.mark.parametrize(
    "candidate",
    [
        "Br c1ccccc1",
        "Br.c1ccccc1",
        "CPc1ccccc1",
        "C[N+](C)(C)C",
        "[13C]O",
        "C/C=C/C",
        "C" * 81,
    ],
)
def test_mbo_validator_enforces_frozen_prompt_constraints(candidate: str) -> None:
    with pytest.raises(CandidateValidationError):
        validate_mbo(candidate)


def test_mbo_validator_keeps_allowed_neutral_fused_scaffold() -> None:
    assert validate_mbo("c1ccc2[nH]ccc2c1").canonical == "c1ccc2[nH]ccc2c1"


def test_outcome_blind_mbo_recovery_stream_is_valid_and_seed_deterministic() -> None:
    seed_zero = recovery_stream(task="MBO", seed=0)
    assert CONTRACT_ID == "outcome_blind_structural_recovery_v2"
    assert seed_zero == recovery_stream(task="MBO", seed=0)
    assert seed_zero != recovery_stream(task="MBO", seed=1)
    canonical = [validate_mbo(candidate).canonical for candidate in seed_zero]
    assert len(seed_zero) >= 48
    assert len(canonical) == len(set(canonical))


@pytest.mark.parametrize("task", ["NHO", "SPO", "TMC"])
def test_outcome_blind_recovery_stream_covers_other_released_tasks(
    task: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(
        str(ROOT / "official_sources/PiEvo_matched_qwen235b/surrogates/envs")
    )
    stream = recovery_stream(task=task, seed=0)
    assert len(stream) >= 24
    assert len(stream[:24]) == len(set(stream[:24]))


def test_successful_evaluator_rows_are_used_for_recovery_deduplication() -> None:
    source = (ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/agents/experiment.py").read_text()
    assert "row.get(\"success\") is True" in source
    assert "recovered candidate may not be materialized" in source


def test_mbo_applicability_domain_rejects_long_chain_extrapolation() -> None:
    accepted = [
        assess_mbo_applicability(candidate)
        for candidate in ("c1ccccc1", "CCO", "CCCC")
    ]
    rejected = assess_mbo_applicability(
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCc1ccccc1"
    )
    assert all(result["accepted"] is True for result in accepted)
    assert all(result["contract_id"] == "mbo_structural_envelope_v2" for result in accepted)
    assert rejected["accepted"] is False
    assert "rotatable_bonds" in rejected["violations"]
    assert "logp" in rejected["violations"]
    assert "fraction_csp3" not in rejected["thresholds"]
    assert "rings" not in rejected["thresholds"]


def test_mbo_applicability_domain_remains_before_the_surrogate_call() -> None:
    for repository in ("PiEvo_matched_qwen235b", "PiEvo_radp_qwen235b"):
        source = (
            ROOT / "official_sources" / repository / "pievo/tools/tool_chembl_official.py"
        ).read_text()
        assert source.index("ad_contract = os.getenv") < source.index("raw_output = float(_predictor().predict(canonical))")


def test_baseline_runner_exports_experiment_root_for_applicability_domain() -> None:
    source = (ROOT / "experiments/baseline_table1_runner.py").read_text()
    assert 'os.environ["PIEVO_EXPERIMENT_ROOT"] = str(ROOT)' in source


def test_baseline_client_timeout_is_bounded() -> None:
    source = (ROOT / "experiments/baseline_table1_runner.py").read_text()
    assert "max_retries=1" in source
    assert "timeout=150.0" in source


def test_warmup_selection_scans_the_full_candidate_pool() -> None:
    class Model:
        n_observations = 1

        def predict(self, features):
            return 0.0, float(features[0])

    class Strategy:
        principles = {"p": "test"}
        gp_models = {"p": Model()}

        @staticmethod
        def _extract_features(candidate, principle_id):
            del principle_id
            return np.array({"first": [1.0], "best": [9.0]}[candidate])

    selected = asyncio.run(
        PiEvo._select_hypothesis_warm_up(Strategy(), ["first", "best"])
    )
    assert selected == "best"


def test_mbo_dynamic_prompt_repeats_the_hard_validity_gate(monkeypatch) -> None:
    monkeypatch.setenv("HIGH_CONFIDENCE_THRESHOLD", "0.8")
    monkeypatch.setenv("NUM_EXPLORATION_CASES", "3")
    _, prompt = get_hypothesis_guidance_prompt(
        task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
        top_principle_text="test principle",
        second_best_principle_text="",
        principle_beliefs={"p": 1.0},
        num_principles=1,
    )
    assert "MBO CANDIDATE VALIDITY [HARD GATE]" in prompt
    assert "Return exactly THREE candidates" in prompt
    assert "RDKit-parseable" in prompt
    assert "speculative fused/spiro ring-closure syntax" in prompt
    assert 'global "each digit appears twice" rule' in prompt
    assert "C1CCCCCCCCCCC1" in prompt
    assert "C1C2CC3CC1CC(C2)C3" in prompt

    system_prompt = (
        ROOT / "experiments/configs/official_pievo_mbo_model_short_smiles.yaml"
    ).read_text(encoding="utf-8")
    assert "every ring digit 1 or 2 appears exactly twice" not in system_prompt
    assert "saturated macrocycle" in system_prompt


def test_authorized_seed0_invalid_feedback_is_bounded_and_actionable(
    tmp_path: Path, monkeypatch
) -> None:
    invalid_candidates = [
        "c1ccc2c3c1cccc3ccc2",
        "Cc1ccc2c3c1cccc3ccc2",
        "O=C1c2ccccc2C(=O)c2cccc12",
        "Oc1ccc2c3c1cccc3ccc2",
        "Clc1ccc2c3c1cccc3ccc2",
        "COc1ccc2c3c1cccc3ccc2CN",
        "Fc1ccc2c3c1cccc3ccc2F",
        "c1ccc2c(c1)c3ccccc3c2",
        "c1ccc2c(c1)c3c(cc1)ccc3cc2",
    ]
    rejection_reasons = []
    for candidate in invalid_candidates:
        with pytest.raises(CandidateValidationError) as exc_info:
            validate_mbo(candidate)
        rejection_reasons.append(str(exc_info.value))
    assert all("cannot be kekulized" in reason for reason in rejection_reasons[:-1])
    assert "ring-closure syntax cannot be parsed" in rejection_reasons[-1]

    audit_path = tmp_path / "candidate_prefilter_audit.jsonl"
    audit_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "candidate": candidate,
                    "candidate_key": candidate,
                    "disposition": "invalid",
                    "reason": "invalid SMILES",
                    "output": 999.0,
                }
            )
            for candidate in invalid_candidates
        )
        + "\n",
        encoding="utf-8",
    )
    feedback = load_mbo_candidate_rejections(tmp_path)
    assert len(feedback) == MBO_REJECTION_FEEDBACK_LIMIT
    assert [row["candidate"] for row in feedback] == invalid_candidates[-6:]

    monkeypatch.setenv("HIGH_CONFIDENCE_THRESHOLD", "0.8")
    monkeypatch.setenv("NUM_EXPLORATION_CASES", "3")
    _, prompt = get_hypothesis_guidance_prompt(
        task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
        top_principle_text="test principle",
        second_best_principle_text="",
        principle_beliefs={"p": 1.0},
        num_principles=1,
        candidate_rejections=feedback,
    )
    assert "RECENT LOCAL VALIDATOR FEEDBACK [SYNTAX ONLY]" in prompt
    assert invalid_candidates[-1] in prompt
    assert f'- Rejected "{invalid_candidates[0]}":' not in prompt
    assert "999" not in prompt

    assert all(
        validate_mbo(candidate).canonical
        for _, candidate in MBO_VALIDATED_SCAFFOLDS
    )
    protocol = json.loads(
        (ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json").read_text()
    )
    config = protocol["minimal_controlled_api_screen"]["candidate_prefilter"]
    assert config["syntax_feedback_enabled"] is True
    assert config["syntax_feedback_limit"] == MBO_REJECTION_FEEDBACK_LIMIT
    assert config["validated_scaffold_count"] == len(MBO_VALIDATED_SCAFFOLDS)


def test_matched_candidate_generation_files_are_byte_identical() -> None:
    for relative in (
        "pievo/agents/abstract.py",
        "pievo/agents/experiment.py",
        "pievo/tools/candidate_validation.py",
        "pievo/word/instruction.py",
    ):
        matched = ROOT / "official_sources/PiEvo_matched_qwen235b" / relative
        radp = ROOT / "official_sources/PiEvo_radp_qwen235b" / relative
        assert matched.read_bytes() == radp.read_bytes()


def test_manifest_summary_keeps_prefilter_rejections_visible(tmp_path: Path) -> None:
    (tmp_path / "official_chembl_tool_audit.jsonl").write_text(
        json.dumps(
            {
                "tool_name": "evaluate_chembl",
                "success": True,
                "output": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "candidate_prefilter_audit.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"disposition": "invalid"}),
                json.dumps({"disposition": "canonical_duplicate"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = audit_summary(tmp_path, "MBO")
    assert summary["primary"] == 1
    assert summary["prefilter_invalid"] == 1
    assert summary["prefilter_duplicate"] == 1
    assert summary["effective_invalid"] == 1
    assert summary["effective_duplicate"] == 1
