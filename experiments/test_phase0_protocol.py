"""Offline regression tests for the official-run Phase 0 boundary.

The project environment does not bundle pytest, so these tests intentionally
use the standard library ``unittest`` runner and never call an LLM or network
service.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official_sources/PiEvo_matched_qwen235b"))

from pievo.group.evolveflow import PiEvo  # noqa: E402
from pievo.group.manage import PrimaryBudgetTermination  # noqa: E402

sys.path.insert(0, str(ROOT))
from experiments.radp_search_online_run_audit import trajectory_metrics  # noqa: E402


class Phase0ProtocolTests(unittest.TestCase):
    def test_final_tool_only_primary_captures_usage_and_token_ceiling(self) -> None:
        class Client:
            def total_usage(self):
                return SimpleNamespace(prompt_tokens=10, completion_tokens=2)

        with tempfile.TemporaryDirectory() as directory:
            old_ceiling = os.environ.get("PIEVO_MAX_TOTAL_TOKENS")
            old_stop = os.environ.get("PIEVO_STOP_REQUESTED")
            try:
                os.environ["WORKING_RESULT_DIR"] = directory
                os.environ["PIEVO_MAX_TOTAL_TOKENS"] = "10"
                os.environ["PIEVO_STOP_REQUESTED"] = "0"
                Path(directory, "official_chembl_tool_audit.jsonl").write_text(
                    json.dumps(
                        {
                            "tool_name": "evaluate_chembl",
                            "success": True,
                            "call_index": 1,
                            "input": "CCO",
                            "output": 1.0,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                pievo = object.__new__(PiEvo)
                pievo.output_dir = directory
                pievo.output_file = str(Path(directory, "submissions.json"))
                pievo.task = "MBO"
                pievo.submissions = []
                pievo.util_client = Client()
                pievo.extra_clients = {}
                self.assertEqual(pievo.reconcile_primary_audit(), 1)
                self.assertEqual(pievo.submissions[0]["usage_total"], 12)
                self.assertEqual(os.environ["PIEVO_STOP_REQUESTED"], "1")
                stop = json.loads(
                    Path(directory, "token_ceiling_stop.json").read_text(encoding="utf-8")
                )
                self.assertEqual(stop["observed_total"], 12)
            finally:
                if old_ceiling is None:
                    os.environ.pop("PIEVO_MAX_TOTAL_TOKENS", None)
                else:
                    os.environ["PIEVO_MAX_TOTAL_TOKENS"] = old_ceiling
                if old_stop is None:
                    os.environ.pop("PIEVO_STOP_REQUESTED", None)
                else:
                    os.environ["PIEVO_STOP_REQUESTED"] = old_stop

    def test_primary_termination_counts_only_admitted_rows(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["WORKING_RESULT_DIR"] = directory
                os.environ["PIEVO_MAX_PRIMARY_EVALS"] = "2"
                os.environ["PIEVO_STOP_REQUESTED"] = "0"
                rows = [
                    {"tool_name": "evaluate_chembl", "success": True, "call_index": 1},
                    {"tool_name": "evaluate_chembl", "success": True, "call_index": 2},
                    {
                        "tool_name": "evaluate_chembl",
                        "success": True,
                        "call_index": 2,
                        "duplicate_suppressed": True,
                    },
                ]
                Path(directory, "official_chembl_tool_audit.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )
                condition = PrimaryBudgetTermination()
                self.assertIsNotNone(await condition([]))
                self.assertTrue(condition.terminated)

        asyncio.run(run())

    def test_audit_reconciliation_fills_tool_only_final_row_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["WORKING_RESULT_DIR"] = directory
            audit_rows = [
                {
                    "tool_name": "evaluate_chembl",
                    "success": True,
                    "call_index": 1,
                    "input": "A",
                    "output": 1.0,
                },
                {
                    "tool_name": "evaluate_chembl",
                    "success": True,
                    "call_index": 2,
                    "input": "B",
                    "output": 2.0,
                },
            ]
            Path(directory, "official_chembl_tool_audit.jsonl").write_text(
                "\n".join(json.dumps(row) for row in audit_rows) + "\n",
                encoding="utf-8",
            )
            pievo = object.__new__(PiEvo)
            pievo.output_dir = directory
            pievo.output_file = str(Path(directory, "submissions.json"))
            pievo.submissions = [
                {"source_agent": "experiment", "json_data": {"candidate": "A", "outcome": 1.0}}
            ]
            self.assertEqual(pievo.reconcile_primary_audit(), 1)
            self.assertEqual(pievo.reconcile_primary_audit(), 0)
            experiment_rows = json.loads(
                Path(directory, "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(experiment_rows), 2)
            self.assertEqual(experiment_rows[-1]["candidate"], "B")

    def test_non_mbo_audits_are_counted_and_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["WORKING_RESULT_DIR"] = directory
            os.environ["PIEVO_MAX_PRIMARY_EVALS"] = "3"
            audit_specs = {
                "official_nho_tool_audit.jsonl": (
                    "evaluate_nanohelix",
                    "fiber_radius=30,helix_radius=60,n_turns=6,pitch=120,wavelength=450,x_y_z=2,direction=0",
                    1.2,
                ),
                "official_spo_tool_audit.jsonl": (
                    "evaluate_supercon",
                    "Y1Ba2Cu3O7",
                    90.3,
                ),
                "official_tmc_tool_audit.jsonl": (
                    "evaluate_tmc",
                    "Pd_A_B_C_D",
                    197.9,
                ),
            }
            for filename, (tool_name, candidate, output) in audit_specs.items():
                Path(directory, filename).write_text(
                    json.dumps(
                        {
                            "tool_name": tool_name,
                            "record_type": "primary",
                            "success": True,
                            "call_index": 1,
                            "input": candidate,
                            "output": output,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            condition = PrimaryBudgetTermination()
            self.assertTrue(asyncio.run(condition([])).content)

            pievo = object.__new__(PiEvo)
            pievo.output_dir = directory
            pievo.output_file = str(Path(directory, "submissions.json"))
            pievo.submissions = []
            self.assertEqual(pievo.reconcile_primary_audit(), 3)
            self.assertEqual(pievo.reconcile_primary_audit(), 0)
            experiment_rows = json.loads(
                Path(directory, "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(experiment_rows), 3)
            submission_payload = json.loads(
                Path(directory, "submissions.json").read_text(encoding="utf-8")
            )
            provenances = {
                record["json_data"]["outcome_provenance"]
                for record in submission_payload["submissions"]
                if record.get("source_agent") == "experiment"
            }
            self.assertEqual(
                provenances,
                {
                    "official_nho_tool_audit",
                    "official_spo_tool_audit",
                    "official_tmc_tool_audit",
                },
            )

    def test_sq_and_auoc_are_deterministic(self) -> None:
        rows = [{"value": value} for value in (1.0, 2.0, 2.5)]
        expected = trajectory_metrics(rows, 2.0)
        for _ in range(3):
            self.assertEqual(trajectory_metrics(rows, 2.0), expected)
        self.assertAlmostEqual(expected["sq_percent"], 125.0)
        self.assertAlmostEqual(expected["auoc_percent"], 91.6666666667)


if __name__ == "__main__":
    unittest.main()
