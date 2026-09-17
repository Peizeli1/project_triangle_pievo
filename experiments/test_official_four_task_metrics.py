"""Offline tests for schema-v3 four-task eligibility and token accounting."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.formal_runtime_contract import (
    APPROVAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    attestation_is_formal,
    expected_runtime,
)
from experiments.formal_usage import last_cumulative_usage, read_cumulative_usage
from experiments.official_four_task_metrics import gate_decision, run_metrics


TASKS = ("NHO", "MBO", "SPO", "TMC")
METHODS = ("pievo", "radp")


def embedding_attestation() -> dict[str, object]:
    return {
        "path": "/tmp/qwen3-embedding",
        "offline": True,
        "file_hashes": {
            "config.json": "d" * 64,
            "model.safetensors": "e" * 64,
            "tokenizer.json": "f" * 64,
        },
        "passed": True,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(root: Path, task: str, method: str, seed: int, value: float) -> Path:
    run_dir = root / f"matched_qwen235b_{method}_{task.lower()}_seed{seed}_formal24"
    run_dir.mkdir(parents=True)
    audit_names = {
        "NHO": "official_nho_tool_audit.jsonl",
        "MBO": "official_chembl_tool_audit.jsonl",
        "SPO": "official_spo_tool_audit.jsonl",
        "TMC": "official_tmc_tool_audit.jsonl",
    }
    tool_names = {
        "NHO": "evaluate_nanohelix",
        "MBO": "evaluate_chembl",
        "SPO": "evaluate_supercon",
        "TMC": "evaluate_tmc",
    }
    rows = [
        {
            "tool_name": tool_names[task],
            "success": True,
            "record_type": "primary",
            "call_index": index,
            "input": f"candidate-{index}",
            "output": value + index * 0.001,
        }
        for index in range(1, 25)
    ]
    (run_dir / audit_names[task]).write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    runtime = expected_runtime(method, task, seed)
    embedding_model = embedding_attestation()
    code_hashes = {"main": "a" * 64, "candidate_validation": "b" * 64}
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "task": task,
        "seed": seed,
        "expected_runtime": runtime,
        "actual_runtime": runtime,
        "embedding_model": embedding_model,
        "code_hashes": code_hashes,
        "require_formal_ready": True,
        "errors": [],
        "passed": True,
    }
    attestation_path = run_dir / "runtime_attestation.json"
    write_json(attestation_path, attestation)

    approval = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "method": method,
        "task": task,
        "passed": True,
        "errors": [],
        "development_manifest_sha256": "c" * 64,
        "code_hashes": code_hashes,
        "embedding_model": embedding_model,
    }
    approval_path = run_dir / "formal_approval.json"
    write_json(approval_path, approval)
    write_json(
        run_dir / "seed_application.json",
        {
            "schema_version": "pievo_seed_application.v1",
            "seed": seed,
            "algorithm_id": runtime["algorithm_id"],
            "llm_client_seeds": [seed, seed, seed, seed],
            "llm_clients_seeded": True,
            "seed_applied": True,
        },
    )
    write_json(
        run_dir / "token_usage.json",
        [
            {
                "round": 1,
                "usage": {
                    "agent": {
                        "prompt": 100 + seed,
                        "completion": 10,
                        "total": 110 + seed,
                    }
                },
            }
        ],
    )
    if method == "radp":
        (run_dir / "radp_runtime_audit.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "radp_runtime_audit.v1",
                    "event": "runtime_start",
                    "algorithm_id": runtime["algorithm_id"],
                    "mode": "active",
                    "provider": f"official_{task.lower()}",
                    "source_gate_enabled": True,
                    "revision_protocol_enabled": True,
                    "backtrack_enabled": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "matched_qwen235b_run_manifest.v3",
            "status": "completed",
            "run_identity": {
                "task": task,
                "method": method,
                "seed": seed,
                "primary_budget": 24,
                "model": "qwen3-235b-a22b-instruct-2507",
                "track": "matched_qwen235b",
                "stage": "formal",
            },
            "paths": {"formal_approval": str(approval_path.resolve())},
            "hashes": {
                "runtime_attestation_sha256": digest(attestation_path),
                "formal_approval_sha256": digest(approval_path),
                "attested_code_hashes": code_hashes,
            },
            "runtime": {"token_ceiling": 1000},
            "claim_boundary": {"matched_evaluator_assets_verified": True},
        },
    )
    return run_dir


class OfficialFourTaskMetricTests(unittest.TestCase):
    def test_runtime_attestation_allows_optional_verifier_metadata(self) -> None:
        runtime = expected_runtime("pievo", "MBO", 0)
        attestation = {
            "schema_version": SCHEMA_VERSION,
            "method": "pievo",
            "task": "MBO",
            "seed": 0,
            "expected_runtime": runtime,
            "actual_runtime": {**runtime, "independent_verifier": ""},
            "embedding_model": embedding_attestation(),
            "code_hashes": {"main": "a" * 64},
            "require_formal_ready": True,
            "errors": [],
            "passed": True,
        }
        self.assertTrue(
            attestation_is_formal(attestation, method="pievo", task="MBO", seed=0)
        )

    def test_runtime_attestation_allows_non_mbo_prefilter_audit_metadata(self) -> None:
        runtime = expected_runtime("pievo", "NHO", 0)
        attestation = {
            "schema_version": SCHEMA_VERSION,
            "method": "pievo",
            "task": "NHO",
            "seed": 0,
            "expected_runtime": runtime,
            "actual_runtime": {
                **runtime,
                "independent_verifier": "",
                "candidate_prefilter_enabled": False,
                "candidate_prefilter_rejection_limit": 0,
                "candidate_prefilter_empty_pool_limit": 0,
            },
            "embedding_model": embedding_attestation(),
            "code_hashes": {"main": "a" * 64},
            "require_formal_ready": True,
            "errors": [],
            "passed": True,
        }
        self.assertTrue(
            attestation_is_formal(attestation, method="pievo", task="NHO", seed=0)
        )

    def test_schema_v3_matrix_can_pass_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = []
            for method in METHODS:
                for task in TASKS:
                    for seed in (0, 1, 2):
                        run_dir = make_run(root, task, method, seed, 1.2 if method == "radp" else 1.0)
                        metrics.append(run_metrics(task, method, seed, run_dir, 24))
            self.assertTrue(all(row["formal_run_candidate"] for row in metrics))
            decision = gate_decision(metrics, 24, reliability_cost_pareto=True)
            self.assertTrue(decision["gate_pass"])
            self.assertEqual(decision["aggregate"]["sq_difference_bootstrap"]["n"], 12)

    def test_v2_manifest_is_implementation_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = make_run(root, "NHO", "radp", 0, 1.0)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "matched_qwen235b_run_manifest.v2"
            write_json(manifest_path, manifest)
            metrics = run_metrics("NHO", "radp", 0, run_dir, 24)
            self.assertFalse(metrics["formal_run_candidate"])
            self.assertTrue(metrics["legacy_diagnostic_only"])
            self.assertIn("manifest_schema_v3", metrics["implementation_invalid_reasons"])

    def test_prefilter_attempts_are_included_in_reliability_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = make_run(root, "SPO", "pievo", 0, 1.0)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["final_evidence"] = {
                "audit": {
                    "effective_invalid": 2,
                    "effective_duplicate": 1,
                    "prefilter_invalid": 2,
                    "prefilter_duplicate": 1,
                }
            }
            write_json(manifest_path, manifest)
            metrics = run_metrics("SPO", "pievo", 0, run_dir, 24)

            self.assertEqual(metrics["invalid_attempts"], 2)
            self.assertEqual(metrics["duplicate_suppressed"], 1)
            self.assertEqual(metrics["prefilter_invalid"], 2)
            self.assertAlmostEqual(metrics["invalid_rate"], 2 / 26)
            self.assertAlmostEqual(metrics["duplicate_rate"], 1 / 27)

    def test_last_nonempty_cumulative_snapshot_is_used(self) -> None:
        usage = [
            {"usage": {"agent": {"prompt": 10, "completion": 2, "total": 12}}},
            {"usage": {"agent": {"prompt": 20, "completion": 3, "total": 23}}},
            {"usage": {}},
        ]
        self.assertEqual(
            last_cumulative_usage(usage),
            {"prompt": 20, "completion": 3, "total": 23, "snapshot_index": 1},
        )

    def test_historical_matrix_artifact_is_not_active_after_cleanup(self) -> None:
        artifact = Path(__file__).resolve().parent / "results/official_gate_decision.json"
        self.assertFalse(artifact.exists())


if __name__ == "__main__":
    unittest.main()
