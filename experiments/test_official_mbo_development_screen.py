import json
from pathlib import Path

from experiments.audit_official_mbo_development_screen import audit, audit_method
from experiments.formal_runtime_contract import repository_hashes, sha256


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _method_fixture(root: Path, method: str) -> Path:
    directory = root / method
    directory.mkdir()
    primary_rows = []
    for call_index, raw_output in enumerate((0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 1):
        corrupted = call_index in {2, 3}
        primary_rows.append(
            {
                "tool_name": "evaluate_chembl",
                "input": f"candidate-{call_index}",
                "success": True,
                "duplicate_suppressed": False,
                "call_index": call_index,
                "corruption_mode": "one_shot_spike",
                "corruption_applied": corrupted,
                "raw_output": raw_output,
                "output": raw_output + 2.0 if corrupted else raw_output,
            }
        )
    _write_jsonl(directory / "official_chembl_tool_audit.jsonl", primary_rows)
    _write_json(
        directory / "run_manifest.json",
        {
            "status": "completed",
            "runtime": {
                "candidate_prefilter_enabled": True,
                "candidate_prefilter_rejection_limit": 12,
                "candidate_prefilter_empty_pool_limit": 2,
                "guidance_only_context": method == "radp",
            },
            "run_identity": {
                "task": "MBO",
                "method": method,
                "seed": 0,
                "primary_budget": 6,
            },
            "paths": {
                "official_repo": str(
                    ROOT
                    / (
                        "official_sources/PiEvo_radp_qwen235b"
                        if method == "radp"
                        else "official_sources/PiEvo_matched_qwen235b"
                    )
                )
            },
            "final_evidence": {"seed_application_passed": True},
        },
    )
    repository = ROOT / (
        "official_sources/PiEvo_radp_qwen235b"
        if method == "radp"
        else "official_sources/PiEvo_matched_qwen235b"
    )
    _write_json(
        directory / "runtime_attestation.json",
        {"code_hashes": repository_hashes(repository, method)},
    )
    # PiEvo records cumulative usage under per-client dictionaries. The audit
    # must take the last snapshot and recursively sum only that snapshot.
    _write_json(
        directory / "token_usage.json",
        [
            {"usage": {"client_a": {"prompt": 10, "completion": 5}}},
            {
                "usage": {
                    "client_a": {"prompt": 20, "completion": 10},
                    "client_b": {"prompt_tokens": 5, "completion_tokens": 5},
                }
            },
        ],
    )
    if method == "radp":
        _write_jsonl(
            directory / "agent_context_boundary_audit.jsonl",
            [
                {
                    "event": "guidance_only_context_reset",
                    "agent": role,
                    "incoming_count": 2,
                    "blocked_count": 2,
                    "admitted_count": 0,
                }
                for role in ("principle", "hypothesis", "experiment")
            ],
        )
    return directory


def test_source_confusion_audit_uses_nested_tokens_and_raw_reward(tmp_path: Path) -> None:
    pievo = _method_fixture(tmp_path, "pievo")
    radp = _method_fixture(tmp_path, "radp")
    _write_jsonl(
        radp / "radp_runtime_audit.jsonl",
        [
            {
                "event": "runtime_start",
                    "algorithm_id": "radp_search_afe_v8",
                "mode": "active",
                "provider": "official_mbo",
            },
            {"event": "gate_finalized"},
        ],
    )
    _write_jsonl(
        radp / "radp_source_gate_audit.jsonl",
        [
            {
                "hypothesis": f"candidate-{call_index}",
                "gate_terminal": True,
                "source_decision": "replace" if call_index in {2, 3} else "accept",
                "raw_update_allowed": call_index not in {2, 3},
                "update_allowed": call_index not in {2, 3},
                "primary_observation": raw + 2.0 if call_index in {2, 3} else raw,
                "verified_observation": raw,
            }
            for call_index, raw in enumerate((0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 1)
        ],
    )

    result = audit(
        tmp_path,
        seed=0,
        budget=6,
        mode="source_confusion",
        token_ceiling=75_000,
    )

    assert result["passed"] is True
    assert result["methods"]["pievo"]["token_usage"]["total"] == 40
    assert result["methods"]["pievo"]["best_observed_output"] == 2.7
    assert result["methods"]["pievo"]["best_true_output"] == 1.0
    assert result["checks"]["radp_guidance_only_context_enabled"] is True
    assert result["checks"]["radp_context_boundary_audited"] is True


def test_afe_v8_audit_counts_use_the_active_schema_version(tmp_path: Path) -> None:
    radp = _method_fixture(tmp_path, "radp")
    manifest = json.loads((radp / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["runtime"]["radp_search_strategy"] = "afe_v8"
    _write_json(radp / "run_manifest.json", manifest)
    _write_jsonl(
        radp / "radp_candidate_portfolio_audit.jsonl",
        [
            {"schema_version": "radp_afe_portfolio.v8"},
            *[
                {"schema_version": "radp_afe_acquisition.v8"}
                for _ in range(6)
            ],
        ],
    )

    result = audit_method(
        radp,
        method="radp",
        seed=0,
        budget=6,
        mode="clean",
        token_ceiling=75_000,
    )

    assert result["afe_portfolio_audit_count"] == 1
    assert result["afe_acquisition_audit_count"] == 6


def test_source_confusion_audit_fails_if_only_one_injection_is_repaired(
    tmp_path: Path,
) -> None:
    _method_fixture(tmp_path, "pievo")
    radp = _method_fixture(tmp_path, "radp")
    _write_jsonl(
        radp / "radp_runtime_audit.jsonl",
        [
            {
                "event": "runtime_start",
                "algorithm_id": "radp_search_afe_v8",
                "mode": "active",
                "provider": "official_mbo",
            },
            {"event": "gate_finalized"},
        ],
    )
    _write_jsonl(
        radp / "radp_source_gate_audit.jsonl",
        [
            {
                "hypothesis": f"candidate-{call_index}",
                "gate_terminal": True,
                "source_decision": "replace",
                "raw_update_allowed": False,
                "update_allowed": False,
                "primary_observation": raw + 2.0,
                "verified_observation": raw,
            }
            if call_index == 2
            else {
                "hypothesis": f"candidate-{call_index}",
                "gate_terminal": True,
                "source_decision": "accept",
                "raw_update_allowed": True,
                "update_allowed": True,
                "primary_observation": raw,
                "verified_observation": raw,
            }
            for call_index, raw in enumerate((0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 1)
        ],
    )

    result = audit(
        tmp_path,
        seed=0,
        budget=6,
        mode="source_confusion",
        token_ceiling=75_000,
    )

    assert result["passed"] is False
    assert result["checks"]["radp_rejects_both_injected_source_failures"] is False
    assert result["checks"]["radp_repairs_both_injected_source_failures"] is False


def test_source_confusion_audit_fails_on_manifest_seed_mismatch(tmp_path: Path) -> None:
    pievo = _method_fixture(tmp_path, "pievo")
    radp = _method_fixture(tmp_path, "radp")
    manifest = json.loads((pievo / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["run_identity"]["seed"] = 1
    _write_json(pievo / "run_manifest.json", manifest)
    _write_jsonl(
        radp / "radp_runtime_audit.jsonl",
        [
            {
                "event": "runtime_start",
                "algorithm_id": "radp_search_afe_v8",
                "mode": "active",
                "provider": "official_mbo",
            },
            {"event": "gate_finalized"},
        ],
    )
    _write_jsonl(
        radp / "radp_source_gate_audit.jsonl",
        [
            {
                "hypothesis": f"candidate-{call_index}",
                "gate_terminal": True,
                "source_decision": "replace",
                "raw_update_allowed": False,
                "update_allowed": False,
                "primary_observation": raw + 2.0,
                "verified_observation": raw,
            }
            if call_index in {2, 3}
            else {
                "hypothesis": f"candidate-{call_index}",
                "gate_terminal": True,
                "source_decision": "accept",
                "raw_update_allowed": True,
                "update_allowed": True,
                "primary_observation": raw,
                "verified_observation": raw,
            }
            for call_index, raw in enumerate((0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 1)
        ],
    )

    result = audit(
        tmp_path,
        seed=0,
        budget=6,
        mode="source_confusion",
        token_ceiling=75_000,
    )

    assert result["passed"] is False
    assert result["checks"]["manifest_identities_match_protocol"] is False


def test_prefilter_rejections_remain_part_of_invalid_noninferiority(
    tmp_path: Path,
) -> None:
    _method_fixture(tmp_path, "pievo")
    radp = _method_fixture(tmp_path, "radp")
    _write_jsonl(
        radp / "candidate_prefilter_audit.jsonl",
        [
            {
                "schema_version": "candidate_prefilter_audit.v1",
                "task": "MBO",
                "candidate": "invalid [",
                "candidate_key": "invalid [",
                "disposition": "invalid",
                "reason": "invalid SMILES",
            }
        ],
    )

    result = audit(
        tmp_path,
        seed=0,
        budget=6,
        mode="source_confusion",
        token_ceiling=75_000,
    )

    assert result["methods"]["radp"]["effective_invalid_count"] == 1
    assert result["checks"]["invalid_duplicate_noninferior"] is False


def test_screen_audit_fails_when_attested_code_is_stale(tmp_path: Path) -> None:
    _method_fixture(tmp_path, "pievo")
    radp = _method_fixture(tmp_path, "radp")
    attestation = json.loads(
        (radp / "runtime_attestation.json").read_text(encoding="utf-8")
    )
    attestation["code_hashes"]["evolveflow"] = "0" * 64
    _write_json(radp / "runtime_attestation.json", attestation)

    result = audit(
        tmp_path,
        seed=0,
        budget=6,
        mode="source_confusion",
        token_ceiling=75_000,
    )

    assert result["passed"] is False
    assert result["checks"]["runtime_code_hashes_current"] is False


def test_screen_audit_recomputes_attested_input_hashes(tmp_path: Path) -> None:
    _method_fixture(tmp_path, "pievo")
    _method_fixture(tmp_path, "radp")
    input_paths = {
        "task_config": ROOT / "experiments/configs/official_pievo_mbo_task.yaml",
        "model_config": ROOT / "experiments/configs/official_pievo_mbo_model_short_smiles.yaml",
        "protocol": ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json",
    }
    for method in ("pievo", "radp"):
        directory = tmp_path / method
        manifest = json.loads((directory / "run_manifest.json").read_text())
        manifest["paths"].update({name: str(path) for name, path in input_paths.items()})
        _write_json(directory / "run_manifest.json", manifest)
        repository = ROOT / (
            "official_sources/PiEvo_radp_qwen235b"
            if method == "radp"
            else "official_sources/PiEvo_matched_qwen235b"
        )
        attestation = {
            "code_hashes": repository_hashes(repository, method)
            | {
                f"input_{name}": sha256(path)
                for name, path in input_paths.items()
            }
        }
        _write_json(directory / "runtime_attestation.json", attestation)

    result = audit(
        tmp_path,
        seed=0,
        budget=6,
        mode="clean",
        token_ceiling=75_000,
    )

    assert result["methods"]["pievo"]["runtime_code_hashes_current"] is True
    assert result["methods"]["radp"]["runtime_code_hashes_current"] is True
