"""Pure offline tests for RADP state, candidate constraints, and preflight."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RadpCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = load_module(
            "radp_core_test",
            ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/radp/core.py",
        )
        cls.validation = load_module(
            "candidate_validation_test",
            ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/tools/candidate_validation.py",
        )

    def evidence(self, **overrides):
        values = {
            "evidence_id": "e1",
            "observation_id": "o1",
            "source": "official_mbo",
            "candidate": "CCO",
            "primary_outcome": 1.0,
            "repeat_outcome": 1.0,
            "independent_outcome": 1.0,
            "verification_cost": 1.0,
        }
        values.update(overrides)
        return self.core.VerificationEvidence(**values)

    def test_noop_and_shadow_do_not_mutate_search_memory(self) -> None:
        noop = self.core.RadpController("noop")
        shadow = self.core.RadpController("shadow")
        self.assertFalse(noop.observe_primary("o1", "x", 1.0, "source"))
        self.assertFalse(shadow.observe_primary("o1", "x", 1.0, "source"))
        shadow.adjudicate(self.evidence())
        self.assertEqual(noop.search_memory, [])
        self.assertEqual(shadow.search_memory, [])
        self.assertEqual(len(shadow.verification_memory), 1)

    def test_active_disagreement_backtracks_observation(self) -> None:
        active = self.core.RadpController("active", tolerance=0.01)
        active.observe_primary("o1", "CCO", 1.0, "official_mbo")
        decision = active.adjudicate(
            self.evidence(repeat_outcome=2.0, independent_outcome=2.0)
        )
        self.assertEqual(decision.action, "replace")
        self.assertFalse(active.search_memory[0]["active"])
        self.assertEqual(decision.selected_outcome, 2.0)

    def test_adaptive_consensus_can_recover_a_noisy_primary(self) -> None:
        active = self.core.RadpController("active", tolerance=0.05)
        active.observe_primary("o1", "CCO", 1.08, "official_mbo")
        decision = active.adjudicate(
            self.evidence(
                primary_outcome=1.08,
                repeat_outcome=0.90,
                independent_outcome=0.91,
                expected_outcome=0.25,
            )
        )
        self.assertTrue(decision.update_allowed)
        self.assertEqual(decision.action, "accept")
        self.assertAlmostEqual(decision.selected_outcome, 0.905)

    def test_missing_verifier_and_exhausted_budget_fail_closed(self) -> None:
        no_verifier = self.core.RadpController("active")
        no_verifier.observe_primary("o1", "CCO", 1.0, "official_mbo")
        decision = no_verifier.adjudicate(
            self.evidence(repeat_outcome=None, independent_outcome=None)
        )
        self.assertEqual(decision.action, "quarantine")
        self.assertFalse(decision.update_allowed)
        self.assertFalse(no_verifier.search_memory[0]["active"])

        exhausted = self.core.RadpController("active", verification_budget=0)
        exhausted.observe_primary("o1", "CCO", 1.0, "official_mbo")
        decision = exhausted.adjudicate(self.evidence(verification_cost=1.0))
        self.assertEqual(decision.action, "stop")
        self.assertFalse(decision.update_allowed)
        self.assertFalse(exhausted.search_memory[0]["active"])

    def test_exact_replays_are_idempotent_but_conflicts_fail(self) -> None:
        active = self.core.RadpController("active")
        self.assertTrue(active.observe_primary("o1", "CCO", 1.0, "official_mbo"))
        self.assertFalse(active.observe_primary("o1", "CCO", 1.0, "official_mbo"))
        with self.assertRaises(ValueError):
            active.observe_primary("o1", "CCN", 2.0, "official_mbo")
        first = active.propose_principle("p1", "text")
        self.assertIs(active.propose_principle("p1", "text"), first)
        with self.assertRaises(ValueError):
            active.propose_principle("p1", "changed")

    def test_principle_requires_independent_and_heldout_support(self) -> None:
        active = self.core.RadpController("active", principle_accept_evidence=2)
        active.propose_principle("p1", "test")
        first = self.evidence(evidence_id="e1", independent_channel=True, heldout_probe=True)
        second = self.evidence(evidence_id="e2", independent_channel=True, heldout_probe=True)
        active.verification_memory.extend([first, second])
        self.assertEqual(active.review_principle("p1", first, supports=True).value, "provisional")
        self.assertEqual(active.review_principle("p1", second, supports=True).value, "accepted")
        weak_contradiction = self.evidence(evidence_id="e3", heldout_probe=False)
        self.assertEqual(
            active.review_principle("p1", weak_contradiction, supports=False).value,
            "accepted",
        )
        contradicted = self.evidence(evidence_id="e4", heldout_probe=True)
        self.assertEqual(active.review_principle("p1", contradicted, supports=False).value, "retracted")

    def test_cost_aware_action_selection_has_a_stop_action(self) -> None:
        active = self.core.RadpController("active", verification_budget=1)
        self.assertEqual(
            active.choose_action(
                expected_search_gain=0.0,
                expected_verification_gain=0.2,
                verification_cost=1.0,
                remaining_primary_budget=3,
            ),
            "STOP",
        )

    def test_candidate_validation_rejects_invalid_values_before_evaluation(self) -> None:
        valid_nho = "fiber_radius=20,helix_radius=30,n_turns=3,pitch=60,wavelength=400,x_y_z=0,direction=1"
        self.assertEqual(self.validation.validate_nho(valid_nho).task, "NHO")
        with self.assertRaises(self.validation.CandidateValidationError):
            self.validation.validate_nho(valid_nho.replace("fiber_radius=20", "fiber_radius=2"))
        with self.assertRaises(self.validation.CandidateValidationError):
            self.validation.validate_spo("Cu0O2")
        pool = {"a": {"charge": -1}, "b": {"charge": 0}}
        self.assertEqual(
            self.validation.validate_tmc("Pd_a_a_b_b", ligand_pool=pool).task,
            "TMC",
        )
        with self.assertRaises(self.validation.CandidateValidationError):
            self.validation.validate_tmc("Pd_a_b_b_b", ligand_pool=pool)

    def test_invalid_smiles_never_reaches_surrogate_predictor(self) -> None:
        matched_repo = ROOT / "official_sources/PiEvo_matched_qwen235b"
        sys.path.insert(0, str(matched_repo))
        try:
            from pievo.tools import tool_chembl_official

            class FailingPredictor:
                def predict(self, value):
                    raise AssertionError("invalid candidate reached predictor")

            tool_chembl_official._PREDICTOR = FailingPredictor()
            result = tool_chembl_official.evaluate_chembl("not a smiles [")
            self.assertFalse(result["success"])
            self.assertEqual(result["error"], "invalid SMILES")
        finally:
            sys.path.remove(str(matched_repo))

    def test_development_dry_run_never_needs_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry"
            environment = os.environ.copy()
            environment.pop("ZJU_TASK_API_KEY", None)
            environment.pop("OPENAI_API_KEY", None)
            environment.update(
                {
                    "PIEVO_TASK": "NHO",
                    "PIEVO_METHOD": "pievo",
                    "PIEVO_RUN_STAGE": "development",
                    "PIEVO_BUDGET": "6",
                    "PIEVO_DRY_RUN": "1",
                    "PIEVO_OUT": str(output),
                }
            )
            result = subprocess.run(
                [str(ROOT / "experiments/run_matched_qwen235b_task.sh")],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("no API key was read", result.stdout)

    def test_paid_screen_checks_key_before_starting_a_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "screen"
            environment = os.environ.copy()
            environment.pop("ZJU_TASK_API_KEY", None)
            environment.pop("OPENAI_API_KEY", None)
            environment["ZJU_TASK_API_KEY_FILE"] = str(Path(directory) / "missing-key")
            environment.update(
                {
                    "PIEVO_DRY_RUN": "0",
                    "PIEVO_MINIMAL_MODE": "source_confusion",
                    "PIEVO_MINIMAL_SCREEN_APPROVAL": "YES",
                    "PIEVO_MINIMAL_OUT_ROOT": str(output),
                }
            )
            result = subprocess.run(
                [str(ROOT / "experiments/run_official_mbo_development_screen.sh")],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Set ZJU_TASK_API_KEY", result.stderr)
            self.assertTrue(output.is_dir())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "minimal_radp_offline_gate.json",
                    "radp_afe_v8_preflight.json",
                },
            )

    def test_development_screen_rejects_unfrozen_radp_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "screen"
            environment = os.environ.copy()
            environment.update(
                {
                    "PIEVO_DRY_RUN": "1",
                    "PIEVO_MINIMAL_MODE": "source_confusion",
                    "PIEVO_MINIMAL_OUT_ROOT": str(output),
                    "PIEVO_RADP_SEARCH_STRATEGY": "legacy_v1",
                }
            )
            result = subprocess.run(
                [str(ROOT / "experiments/run_official_mbo_development_screen.sh")],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertIn("frozen to RADP strategy afe_v8", result.stderr)

    def test_formal_radp_contract_requires_active_v9_strategy(self) -> None:
        contract = load_module(
            "formal_runtime_contract_test",
            ROOT / "experiments/formal_runtime_contract.py",
        )
        environment = {
            "PIEVO_ALGORITHM_ID": "radp_search_active_v1",
            "PIEVO_SOURCE_GATE": "1",
            "PIEVO_SOURCE_GATE_MODE": "active",
            "PIEVO_SOURCE_GATE_PROVIDER": "official_mbo",
            "PIEVO_REVISION_PROTOCOL": "1",
            "PIEVO_ACTIVE_BACKTRACK": "1",
            "PIEVO_GUIDANCE_ONLY_CONTEXT": "1",
            "PIEVO_RADP_SEARCH_STRATEGY": "legacy_v1",
            "PIEVO_SEED": "0",
        }
        result = contract.validate_runtime(
            method="radp",
            task="MBO",
            seed=0,
            repository=ROOT / "official_sources/PiEvo_radp_qwen235b",
            environment=environment,
            input_paths={
                "task_config": ROOT / "experiments/configs/official_pievo_mbo_task.yaml",
                "model_config": ROOT / "experiments/configs/official_pievo_mbo_model_short_smiles.yaml",
                "protocol": ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json",
            },
            require_formal_ready=True,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("frozen afe_v9 acquisition strategy" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
