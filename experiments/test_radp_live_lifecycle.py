"""Offline proof that RADP lifecycle transitions mutate PiEvo's live state."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pievo.group.evolveflow import PiEvo
from pievo.radp import GateMode, RadpRuntime, VerificationEvidence


class LiveLifecycleTests(unittest.TestCase):
    def test_accepted_principle_backtrack_removes_all_live_state_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pievo = object.__new__(PiEvo)
            pievo.output_dir = directory
            pievo._revision_protocol_enabled = True
            pievo._radp_runtime = RadpRuntime(
                algorithm_id="radp_search_afe_v8",
                mode=GateMode.ACTIVE,
                provider="controlled",
                output_dir=directory,
                tolerance=1e-6,
                verification_budget=20,
            )
            pievo.principles = {"base": "base", "revision": "revision"}
            pievo.principles_rationals = {"revision": {"why": "test"}}
            pievo.principle_priors = {"base": 0.5, "revision": 0.5}
            pievo.principle_beliefs = {"base": 0.4, "revision": 0.6}
            pievo.gp_models = {"base": object(), "revision": object()}
            pievo.prediction_cache = {("x", "revision"): 1.0}
            pievo.optimal_prediction_cache = {"revision": ("x", 1.0)}
            pievo.feature_cache = {("x", "revision"): object()}
            pievo._retracted_principle_ids = set()
            pievo.round_by_PHE_order_before_P = 2

            pievo._radp_runtime.controller.propose_principle("base", "base")
            pievo._radp_runtime.propose_principle(
                principle_id="revision",
                text="revision",
                parent_id="base",
            )

            def evidence(identifier: str, *, heldout: float) -> VerificationEvidence:
                return VerificationEvidence(
                    evidence_id=identifier,
                    observation_id=identifier,
                    source="controlled",
                    candidate="revision",
                    primary_outcome=1.0,
                    independent_outcome=1.0,
                    heldout_outcome=heldout,
                    verification_cost=1.0,
                    independent_channel=True,
                    heldout_probe=True,
                )

            self.assertEqual(
                pievo._apply_principle_review(
                    principle_id="revision",
                    evidence=evidence("support_1", heldout=1.0),
                    supports=True,
                ),
                "provisional",
            )
            with self.assertRaises(ValueError):
                pievo._apply_principle_review(
                    principle_id="revision",
                    evidence=evidence("support_1", heldout=0.0),
                    supports=False,
                )
            self.assertEqual(
                pievo._apply_principle_review(
                    principle_id="revision",
                    evidence=evidence("support_2", heldout=1.0),
                    supports=True,
                ),
                "accepted",
            )
            self.assertEqual(
                pievo._apply_principle_review(
                    principle_id="revision",
                    evidence=evidence("contradiction", heldout=0.0),
                    supports=False,
                ),
                "retracted",
            )
            self.assertNotIn("revision", pievo.principles)
            self.assertNotIn("revision", pievo.principles_rationals)
            self.assertNotIn("revision", pievo.principle_priors)
            self.assertNotIn("revision", pievo.principle_beliefs)
            self.assertNotIn("revision", pievo.gp_models)
            self.assertEqual(pievo.principle_beliefs, {"base": 1.0})
            self.assertEqual(
                pievo._backtrack_principles("revision", reason="replay"),
                [],
            )

            rows = [
                json.loads(line)
                for line in Path(directory, "radp_runtime_audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            transitions = [row for row in rows if row.get("event") == "principle_transition"]
            self.assertEqual(transitions[-1]["to_status"], "retracted")
            self.assertEqual(
                sum(row.get("event") == "gp_backtrack" for row in rows),
                1,
            )


if __name__ == "__main__":
    unittest.main()
