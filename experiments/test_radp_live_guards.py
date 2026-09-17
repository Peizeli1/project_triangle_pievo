"""Regression tests for the RADP group-loop guards.

These tests deliberately avoid model clients and evaluator calls.  They cover
the two lifecycle edges that ordinary core tests cannot see: replayed agent
messages and the end-of-run gate flush.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallSummaryMessage,
)
from autogen_core.models import FunctionExecutionResult

from pievo.group import evolveflow
from pievo.group.manage import SubmissionBasedGroupChat
from pievo.agents.abstract import (
    Agent,
    _audit_context_boundary,
    configured_model_attempts,
    guidance_only_context_enabled,
)
from pievo.group.evolveflow import PiEvo
from pievo.word.instruction import get_hypothesis_guidance_prompt


class _RuntimeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def audit(self, event: str, payload: dict[str, object]) -> None:
        self.events.append((event, payload))


class _StatusRuntime:
    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses

    def principle_status(self, principle_id: str) -> str | None:
        return self.statuses.get(principle_id)


class LiveLoopGuardTests(unittest.TestCase):
    def test_guidance_only_context_covers_all_model_roles(self) -> None:
        with patch.dict(
            os.environ, {"PIEVO_GUIDANCE_ONLY_CONTEXT": "1"}, clear=False
        ):
            self.assertTrue(guidance_only_context_enabled("principle"))
            self.assertTrue(guidance_only_context_enabled("hypothesis"))
            self.assertTrue(guidance_only_context_enabled("experiment"))
            self.assertFalse(guidance_only_context_enabled("user_proxy"))

        with patch.dict(
            os.environ, {"PIEVO_GUIDANCE_ONLY_CONTEXT": "0"}, clear=False
        ):
            self.assertFalse(guidance_only_context_enabled("hypothesis"))

    def test_context_boundary_audit_records_counts_without_raw_content(self) -> None:
        class Message:
            def __init__(self, source: str, content: str) -> None:
                self.source = source
                self.content = content

        messages = [
            Message("experiment", "raw injected outcome 99.0"),
            Message("principle", "unaccepted private principle"),
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"WORKING_RESULT_DIR": directory}, clear=False
        ):
            _audit_context_boundary("hypothesis", messages)
            rows = [
                json.loads(line)
                for line in (
                    Path(directory) / "agent_context_boundary_audit.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(rows[0]["blocked_count"], 2)
        self.assertEqual(rows[0]["admitted_count"], 0)
        self.assertEqual(rows[0]["incoming_sources"], ["experiment", "principle"])
        self.assertNotIn("99.0", json.dumps(rows[0]))
        self.assertNotIn("private principle", json.dumps(rows[0]))

    def test_context_is_cleared_before_sanitized_guidance_is_added(self) -> None:
        source = inspect.getsource(Agent.on_messages_stream)
        self.assertLess(
            source.index("await model_context.clear()"),
            source.index("content=await self.get_pievo_guidance()"),
        )
        self.assertIn("() if guidance_only_context else messages", source)
        self.assertIn("if not guidance_only_context:", source)

    def test_outer_model_retry_respects_zero_retry_budget(self) -> None:
        with patch.dict(os.environ, {"PIEVO_MAX_RETRIES": "0"}, clear=False):
            self.assertEqual(configured_model_attempts(), 1)
        with patch.dict(os.environ, {"PIEVO_MAX_RETRIES": "2"}, clear=False):
            self.assertEqual(configured_model_attempts(), 3)
        with patch.dict(os.environ, {"PIEVO_MAX_RETRIES": "invalid"}, clear=False):
            self.assertEqual(configured_model_attempts(), 1)

    def test_radp_exhausted_candidate_batch_returns_to_hypothesis_cycle(self) -> None:
        group = object.__new__(SubmissionBasedGroupChat)
        group.tool_call_in_progress = True
        group.active_tool_call_agent = "experiment"
        group.current_speaker_name = "experiment"
        group.current_speaker_index = 2
        group.agent_rotation_order = ["principle", "hypothesis", "experiment"]
        group.agent_map = {name: object() for name in group.agent_rotation_order}
        group.conversation_flow = []
        group.turn_count = 0
        group.last_submission_detected = False
        group._detect_submission_signal = lambda _message: False

        message = ToolCallSummaryMessage(
            source="experiment",
            content="tool result",
            tool_calls=[],
            results=[],
        )
        with patch.dict(
            os.environ,
            {
                "PIEVO_GUIDANCE_ONLY_CONTEXT": "1",
                "PIEVO_CANDIDATE_POOL_REMAINING": "0",
            },
            clear=False,
        ):
            selected = group._handle_tool_call_sequence(message, "experiment")
            remaining = os.environ.get("PIEVO_CANDIDATE_POOL_REMAINING")

        self.assertEqual(selected, "principle")
        self.assertEqual(group.current_speaker_name, "principle")
        self.assertEqual(remaining, "-1")

    def test_radp_nonempty_candidate_batch_keeps_experiment_role(self) -> None:
        group = object.__new__(SubmissionBasedGroupChat)
        group.tool_call_in_progress = True
        group.active_tool_call_agent = "experiment"
        group.current_speaker_name = "experiment"
        group.current_speaker_index = 2
        group.agent_rotation_order = ["principle", "hypothesis", "experiment"]
        group.agent_map = {name: object() for name in group.agent_rotation_order}
        group.conversation_flow = []
        group.turn_count = 0
        group.last_submission_detected = False
        group._detect_submission_signal = lambda _message: False

        message = ToolCallSummaryMessage(
            source="experiment",
            content="tool result",
            tool_calls=[],
            results=[],
        )
        with patch.dict(
            os.environ,
            {
                "PIEVO_GUIDANCE_ONLY_CONTEXT": "1",
                "PIEVO_CANDIDATE_POOL_REMAINING": "1",
            },
            clear=False,
        ):
            selected = group._handle_tool_call_sequence(message, "experiment")

        self.assertEqual(selected, "experiment")
        self.assertEqual(group.current_speaker_name, "experiment")

    def test_tool_summary_materializes_authoritative_primary(self) -> None:
        tool_row = {
            "tool_name": "evaluate_chembl",
            "input": "c1ccccc1",
            "output": 4.25,
            "success": True,
            "call_index": 1,
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"WORKING_RESULT_DIR": directory}, clear=False
        ):
            audit_path = Path(directory) / "official_chembl_tool_audit.jsonl"
            audit_path.write_text(json.dumps(tool_row) + "\n", encoding="utf-8")

            pievo = object.__new__(PiEvo)
            pievo.output_dir = directory
            pievo.submissions = []
            pievo._processed_submission_hashes = set()
            pievo._processed_raw_principle_hashes = set()
            pievo._radp_verified_outcomes = {}
            pievo.principles_rationals = {}
            pievo._current_usage = lambda: {}
            pievo._enforce_token_ceiling = lambda _usage: 0
            pievo.save_to_file = lambda: None

            rows = pievo._process_message(json.dumps(tool_row), "experiment")

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["json_data"],
            {
                "candidate": "c1ccccc1",
                "outcome": 4.25,
                "call_index": 1,
                "outcome_provenance": "official_chembl_tool_audit",
            },
        )

    def test_tool_execution_event_materializes_authoritative_primary(self) -> None:
        tool_row = {
            "tool_name": "evaluate_chembl",
            "input": "c1ccccc1",
            "output": 4.25,
            "success": True,
            "call_index": 1,
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"WORKING_RESULT_DIR": directory}, clear=False
        ):
            (Path(directory) / "official_chembl_tool_audit.jsonl").write_text(
                json.dumps(tool_row) + "\n", encoding="utf-8"
            )
            pievo = object.__new__(PiEvo)
            pievo.output_dir = directory
            pievo.submissions = []
            pievo._processed_submission_hashes = set()
            pievo._processed_raw_principle_hashes = set()
            pievo._radp_verified_outcomes = {}
            pievo.principles_rationals = {}
            pievo.is_exploitation_phase = False
            pievo.exploitation_candidates = []
            pievo._current_usage = lambda: {}
            pievo._enforce_token_ceiling = lambda _usage: 0
            pievo.save_to_file = lambda: None
            event = ToolCallExecutionEvent(
                source="experiment",
                content=[
                    FunctionExecutionResult(
                        content=json.dumps(tool_row),
                        name="evaluate_chembl",
                        call_id="call-1",
                        is_error=False,
                    )
                ],
            )

            source = pievo.gather_submission_from_message([event])

        self.assertEqual(source, "experiment")
        self.assertEqual(len(pievo.submissions), 1)
        self.assertEqual(pievo.submissions[0]["json_data"]["outcome"], 4.25)

    def test_tool_summary_is_ingested_before_selector_routes_next_role(self) -> None:
        class Strategy:
            def __init__(self) -> None:
                self.messages = []

            def gather_submission_from_message(self, messages):
                self.messages.extend(messages)

        class ExperimentAgent:
            def __init__(self) -> None:
                self.strategy = Strategy()

        group = object.__new__(SubmissionBasedGroupChat)
        group.tool_call_in_progress = True
        group.active_tool_call_agent = "experiment"
        group.current_speaker_name = "experiment"
        group.current_speaker_index = 2
        group.agent_rotation_order = ["principle", "hypothesis", "experiment"]
        experiment_agent = ExperimentAgent()
        group.agent_map = {"experiment": experiment_agent}
        group.conversation_flow = []
        group.turn_count = 0
        group.last_submission_detected = False
        group._detect_submission_signal = lambda _message: False

        message = ToolCallSummaryMessage(
            source="experiment",
            content=json.dumps(
                {
                    "tool_name": "evaluate_chembl",
                    "input": "c1ccccc1",
                    "output": 4.25,
                    "success": True,
                }
            ),
            tool_calls=[],
            results=[],
        )
        with patch.dict(
            os.environ,
            {"PIEVO_GUIDANCE_ONLY_CONTEXT": "0", "PIEVO_STOP_REQUESTED": "0"},
            clear=False,
        ):
            selected = group._handle_tool_call_sequence(message, "experiment")

        self.assertEqual(selected, "experiment")
        self.assertEqual(experiment_agent.strategy.messages, [message])

    def test_radp_empty_guidance_text_routes_to_new_hypothesis_cycle(self) -> None:
        group = object.__new__(SubmissionBasedGroupChat)
        group.tool_call_in_progress = False
        group.active_tool_call_agent = None
        group.current_speaker_name = "experiment"
        group.current_speaker_index = 2
        group.agent_rotation_order = ["principle", "hypothesis", "experiment"]
        group.agent_map = {name: object() for name in group.agent_rotation_order}
        group.conversation_flow = []
        group.turn_count = 2
        group.last_submission_detected = False
        group.budget = 6
        group.experiment_count = 1
        group.submission_patterns = {
            "principle": r"PRINCIPLE_SUBMISSION",
            "hypothesis": r"HYPOTHESIS_SUBMISSION",
            "experiment": r"EXPERIMENT_SUBMISSION",
        }

        message = TextMessage(
            source="experiment",
            content="Instruct the Hypothesis Agent to provide diverse candidates.",
        )
        with patch.dict(
            os.environ,
            {
                "PIEVO_GUIDANCE_ONLY_CONTEXT": "1",
                "PIEVO_CANDIDATE_POOL_REMAINING": "0",
            },
            clear=False,
        ):
            selected = group._select_next_speaker([message])

        self.assertEqual(selected, "principle")
        self.assertEqual(group.current_speaker_name, "principle")

    def test_main_reconciles_tool_only_primary_before_gate_finalization(self) -> None:
        main_path = Path(evolveflow.__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text(encoding="utf-8")
        console_index = source.index("await Console(stream, output_stats=True)")
        reconcile_index = source.index(
            "piflow.pievo.reconcile_primary_audit()", console_index
        )
        finalize_index = source.index("await finalize_gate()", reconcile_index)
        self.assertLess(console_index, reconcile_index)
        self.assertLess(reconcile_index, finalize_index)

    def test_replayed_raw_principle_payload_is_consumed_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pievo = object.__new__(PiEvo)
            pievo.output_dir = directory
            pievo.submissions = []
            pievo.principles_rationals = {}
            pievo._processed_submission_hashes = set()
            pievo._processed_raw_principle_hashes = set()
            pievo.is_exploitation_phase = False
            pievo.exploitation_candidates = []
            pievo._current_usage = lambda: {}
            pievo._enforce_token_ceiling = lambda usage: 0
            pievo.save_to_file = lambda: None

            prepared = {"count": 0}

            def prepare(submission):
                prepared["count"] += 1
                submission["json_data"] = {
                    f"principle_revision_{prepared['count']}": {
                        "PRINCIPLE_SUBMISSION": "one revision",
                    }
                }
                return submission

            pievo._prepare_principle_revision_submission = prepare
            raw_submission = {
                "source_agent": "principle",
                "json_data": {
                    "principle_molecular_rule": {
                        "PRINCIPLE_SUBMISSION": "one revision"
                    }
                },
            }

            with patch.object(
                evolveflow,
                "extract_json_blocks",
                side_effect=lambda *_: [copy.deepcopy(raw_submission)],
            ):
                first = pievo._process_message("same model response", "principle")
                second = pievo._process_message("same model response", "principle")

            self.assertEqual(first, [pievo.submissions[0]])
            self.assertEqual(second, [])
            self.assertEqual(prepared["count"], 1)
            self.assertEqual(len(pievo.submissions), 1)

    def test_reconciled_single_result_is_extracted_into_history(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.task = "Molecular Bio-activity Optimization (MBO) maximize pChEMBL"
        pievo._radp_verified_outcomes = {}
        pievo.submissions = [
            {
                "source_agent": "hypothesis",
                "json_data": {
                    "HYPOTHESIS_SUBMISSION_A": {
                        "candidate": "c1ccccc1",
                        "rational": "control",
                    }
                },
            },
            {
                "source_agent": "experiment",
                "json_data": {
                    "candidate": "c1ccccc1",
                    "outcome": 4.25,
                    "audit_reconciled": True,
                },
            },
        ]

        with patch.dict(os.environ, {"MOF_CIF_OUTPUT_DIR": ""}, clear=False):
            pievo._extract_history()

        self.assertEqual(pievo.history, [("c1ccccc1", 4.25)])

    def test_reconciled_audit_primary_without_model_echo_enters_history(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.task = "Molecular Bio-activity Optimization (MBO) maximize pChEMBL"
        pievo._radp_verified_outcomes = {}
        pievo.submissions = [
            {
                "source_agent": "experiment",
                "json_data": {
                    "candidate": "CCO",
                    "outcome": 3.5,
                    "outcome_provenance": "official_chembl_tool_audit",
                    "audit_reconciled": True,
                },
            }
        ]

        pievo._extract_history()

        self.assertEqual(pievo.history, [("CCO", 3.5)])

    def test_source_gate_reconciles_history_before_search(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.history = [("O=C1CCCCCCCCCCC1", 6.068391799926758)]
        pievo._radp_gate_terminal_candidates = set()
        pievo._radp_verified_outcomes = {}
        pievo._radp_quarantined_candidates = set()
        pievo._candidate_key = PiEvo._candidate_key.__get__(pievo, PiEvo)

        def replace(**_: object) -> dict[str, object]:
            return {
                "update_allowed": False,
                "verified_observation": 4.068391799926758,
            }

        pievo._radp_source_gate_decision = replace
        with patch.dict(os.environ, {"PIEVO_SOURCE_GATE": "1"}, clear=False):
            pievo._reconcile_source_gate_before_search()

        self.assertEqual(pievo.history, [("O=C1CCCCCCCCCCC1", 4.068391799926758)])
        self.assertEqual(
            pievo._radp_gate_terminal_candidates,
            {"O=C1CCCCCCCCCCC1"},
        )
        self.assertNotIn(
            6.068391799926758,
            [outcome for _, outcome in pievo.history],
        )

    def test_full_task_text_still_uses_canonical_candidate_keys(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.task = "Molecular Bio-activity Optimization (MBO) maximize pChEMBL"

        self.assertEqual(
            pievo._candidate_key("Cc1ccccc1N(=O)=O"),
            pievo._candidate_key("Cc1ccccc1[N+](=O)[O-]"),
        )

    def test_hypothesis_guidance_can_expose_only_accepted_principles(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.principles = {
            "proposed": "unreviewed rule",
            "accepted": "reviewed rule",
        }
        pievo.principle_beliefs = {"proposed": 0.8, "accepted": 0.2}
        pievo._radp_runtime = _StatusRuntime(
            {"proposed": "provisional", "accepted": "accepted"}
        )

        with patch.dict(os.environ, {"PIEVO_ACCEPTED_PRINCIPLE_ONLY": "1"}):
            principles, beliefs = pievo._hypothesis_guidance_principle_state()

        self.assertEqual(principles, {"accepted": "reviewed rule"})
        self.assertEqual(beliefs, {"accepted": 1.0})

    def test_hypothesis_guidance_has_neutral_context_before_acceptance(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.principles = {"proposed": "unreviewed rule"}
        pievo.principle_beliefs = {"proposed": 1.0}
        pievo._radp_runtime = _StatusRuntime({"proposed": "provisional"})

        with patch.dict(os.environ, {"PIEVO_ACCEPTED_PRINCIPLE_ONLY": "1"}):
            principles, beliefs = pievo._hypothesis_guidance_principle_state()

        self.assertEqual(principles, {})
        self.assertEqual(beliefs, {})

    def test_neutral_bootstrap_prompt_does_not_stress_test_empty_principle(self) -> None:
        with patch.dict(
            os.environ,
            {"PIEVO_ACCEPTED_PRINCIPLE_ONLY": "1"},
            clear=False,
        ):
            guidance_type, guidance = get_hypothesis_guidance_prompt(
                task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
                top_principle_text="",
                second_best_principle_text="",
                principle_beliefs={},
                num_principles=0,
                candidates_done=[],
                principles={},
            )

        self.assertEqual(guidance_type, "NEUTRAL_DIVERSITY")
        self.assertIn("NO ACCEPTED PRINCIPLE YET", guidance)
        self.assertNotIn("DIRECTED PRINCIPLE:\n\"\"", guidance)

    def test_neutral_bootstrap_prompt_hides_evaluator_outcomes(self) -> None:
        with patch.dict(
            os.environ,
            {"PIEVO_ACCEPTED_PRINCIPLE_ONLY": "1"},
            clear=False,
        ):
            guidance_type, guidance = get_hypothesis_guidance_prompt(
                task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
                top_principle_text="",
                second_best_principle_text="",
                principle_beliefs={},
                num_principles=0,
                candidates_done=[
                    {
                        "candidate": "CCO",
                        "outcome": 9.876543,
                        "reasoning": "private evaluator reasoning",
                    }
                ],
                principles={},
            )

        self.assertEqual(guidance_type, "NEUTRAL_DIVERSITY")
        self.assertIn('"CCO"', guidance)
        self.assertNotIn("9.876543", guidance)
        self.assertNotIn("private evaluator reasoning", guidance)

    def test_reward_guided_neutral_prompt_exposes_only_clean_outcome(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PIEVO_ACCEPTED_PRINCIPLE_ONLY": "1",
                "PIEVO_NEUTRAL_REWARD_GUIDED": "1",
            },
            clear=False,
        ):
            guidance_type, guidance = get_hypothesis_guidance_prompt(
                task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
                top_principle_text="",
                second_best_principle_text="",
                principle_beliefs={},
                num_principles=0,
                candidates_done=[
                    {
                        "candidate": "CCO",
                        "outcome": 9.876543,
                        "reasoning": "private evaluator reasoning",
                    }
                ],
                principles={},
            )

        self.assertEqual(guidance_type, "NEUTRAL_DIVERSITY")
        self.assertIn("clean observed outcome: 9.876543", guidance)
        self.assertNotIn("private evaluator reasoning", guidance)
        self.assertIn("do not use verifier or source-gate details", guidance.lower())

    def test_reward_guided_neutral_prompt_requires_ranked_portfolio(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PIEVO_ACCEPTED_PRINCIPLE_ONLY": "1",
                "PIEVO_NEUTRAL_REWARD_GUIDED": "1",
            },
            clear=False,
        ):
            guidance_type, guidance = get_hypothesis_guidance_prompt(
                task="Molecular Bio-activity Optimization (MBO): maximize pChEMBL",
                top_principle_text="",
                second_best_principle_text="",
                principle_beliefs={},
                num_principles=0,
                candidates_done=[
                    {"candidate": "c1ccccc1", "outcome": 2.0},
                    {"candidate": "c1ccc2ccccc2c1", "outcome": 5.0},
                ],
                principles={},
            )

        self.assertEqual(guidance_type, "NEUTRAL_DIVERSITY")
        self.assertIn("CLEAN-REWARD PORTFOLIO", guidance)
        self.assertIn("A is", guidance)
        self.assertIn("A is one local", guidance)
        self.assertIn("B is one high-prior cross-scaffold", guidance)
        self.assertIn("C is one orthogonal exploration", guidance)
        self.assertIn("clean observed outcome: 5.000000", guidance)

    def test_matched_instruction_files_remain_identical(self) -> None:
        root = Path(__file__).resolve().parents[1]
        matched = (
            root / "official_sources/PiEvo_matched_qwen235b/pievo/word/instruction.py"
        )
        radp = (
            root / "official_sources/PiEvo_radp_qwen235b/pievo/word/instruction.py"
        )
        self.assertEqual(matched.read_bytes(), radp.read_bytes())

    def test_reward_guided_selector_prefers_high_reward_neighborhood(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.task = "Molecular Bio-activity Optimization (MBO) maximize pChEMBL"
        pievo.history = [
            ("c1ccccc1", 1.0),
            ("c1ccc2ccccc2c1", 5.0),
        ]
        candidates = [
            "Oc1ccccc1",
            "Oc1ccc2ccccc2c1",
            "CCOC(=O)c1ccccc1",
        ]

        selected = pievo._select_hypothesis_neutral_diversity(
            candidates, reward_guided=True
        )

        self.assertEqual(selected, candidates[1])

    def test_reward_guided_hypothesis_context_uses_reconciled_history(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.task = "Molecular Bio-activity Optimization (MBO): maximize pChEMBL"
        pievo.status = ""
        pievo.principles = {}
        pievo.principle_beliefs = {}
        pievo.history = [("CCO", 3.2)]
        pievo.submissions = [
            {
                "source_agent": "experiment",
                "json_data": {"candidate": "CCO", "outcome": 99.0},
            }
        ]
        pievo.exploitation_candidates = []
        pievo.round_by_PHE_order_before_P = 0
        pievo.is_exploitation_phase = False
        pievo._calculate_principle_uncertainty = lambda: 0.0

        async def noop_update() -> None:
            return None

        pievo.update_pievo_state = noop_update
        pievo._hypothesis_guidance_principle_state = lambda: ({}, {})

        captured: dict[str, object] = {}

        def fake_prompt(**kwargs):
            captured.update(kwargs)
            return "NEUTRAL_DIVERSITY", "guidance"

        with tempfile.TemporaryDirectory() as directory:
            pievo.output_dir = directory
            with patch.dict(
                os.environ,
                {
                    "PIEVO_ACCEPTED_PRINCIPLE_ONLY": "1",
                    "PIEVO_NEUTRAL_REWARD_GUIDED": "1",
                },
                clear=False,
            ), patch.object(
                evolveflow, "get_hypothesis_guidance_prompt", side_effect=fake_prompt
            ):
                asyncio.run(pievo.get_hypothesis_guidance())

        self.assertEqual(captured["candidates_done"], [{"candidate": "CCO", "outcome": 3.2}])

    def test_neutral_selector_is_not_first_candidate_after_history(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.task = "Molecular Bio-activity Optimization (MBO) maximize pChEMBL"
        pievo.history = [("c1ccccc1", 1.0)]
        candidates = [
            "c1ccccc1",
            "c1ccc2ccccc2c1",
            "FC(F)(F)c1ccc2ccccc2c1",
        ]

        selected = pievo._select_hypothesis_neutral_diversity(candidates)

        self.assertNotEqual(selected, candidates[0])

    def test_neutral_selector_matches_ordered_warmup_without_history(self) -> None:
        pievo = object.__new__(PiEvo)
        pievo.task = "Molecular Bio-activity Optimization (MBO) maximize pChEMBL"
        pievo.history = []
        candidates = [
            "COc1ccc2ccccc2c1",
            "N#Cc1ccc2ccccc2c1",
            "Fc1ccc2ccccc2c1",
        ]

        selected = pievo._select_hypothesis_neutral_diversity(candidates)

        self.assertEqual(selected, candidates[0])

    def test_finalize_radp_gate_flushes_and_records_terminal_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pievo = object.__new__(PiEvo)
            pievo._radp_gate_terminal_candidates = {"a", "b", "c"}
            pievo._radp_verification_calls = 3
            pievo._radp_runtime = _RuntimeRecorder()
            calls: list[str] = []

            async def refresh() -> None:
                calls.append("refresh")

            async def detect(*, force_terminal: bool = False) -> None:
                calls.append(f"detect:{force_terminal}")

            pievo.update_pievo_state = refresh
            pievo._detect_anomalies = detect

            with patch.dict(os.environ, {"PIEVO_SOURCE_GATE": "1"}, clear=False):
                asyncio.run(pievo.finalize_radp_gate())

            self.assertEqual(calls, ["refresh", "detect:True"])
            self.assertEqual(
                pievo._radp_runtime.events,
                [
                    (
                        "gate_finalized",
                        {"terminal_candidate_count": 3, "verification_calls": 3},
                    )
                ],
            )

    def test_terminal_detector_adjudicates_clean_negative_tool_only_primary(self) -> None:
        """A final negative evaluator value still receives one terminal gate."""
        pievo = object.__new__(PiEvo)
        pievo.sigma = 0.1
        pievo.history = [("final-tool-only", -1.0)]
        pievo.principles = {"p0": "initial"}
        pievo.principle_beliefs = {"p0": 1.0}
        pievo.gp_models = {
            "p0": type(
                "Model",
                (),
                {
                    "n_observations": 1,
                    "predict": lambda self, features: (0.0, 1.0),
                },
            )()
        }
        pievo._radp_gate_terminal_candidates = set()
        pievo._candidate_key = PiEvo._candidate_key.__get__(pievo, PiEvo)
        pievo._calculate_adaptive_anomaly_threshold = lambda: 0.99
        pievo._extract_features = lambda hypothesis, principle_id: [0.0]
        pievo._normalize_y = lambda value: float(value)
        pievo._denormalize_y = lambda value: float(value)
        gate_calls: list[dict[str, object]] = []

        def gate(**kwargs: object) -> dict[str, object]:
            gate_calls.append(kwargs)
            return {"update_allowed": True}

        pievo._radp_source_gate_decision = gate

        with patch.dict(
            os.environ,
            {
                "PIEVO_SOURCE_GATE_SCORE_GE": "",
            },
            clear=False,
        ):
            anomalies = asyncio.run(pievo._detect_anomalies(force_terminal=True))

        self.assertEqual(anomalies, [])
        self.assertEqual(pievo._radp_gate_terminal_candidates, {"final-tool-only"})
        self.assertEqual(len(gate_calls), 1)
        self.assertEqual(gate_calls[0]["trigger"], "terminal_reconciliation")


if __name__ == "__main__":
    unittest.main()
