#!/usr/bin/env python3
"""Run the offline gate before any four-task formal API collection.

This check validates the frozen protocol, method-specific paths, runner,
configs, imports, and the published-asset boundary.  It deliberately never
starts PiEvo and never contacts an LLM service.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from formal_runtime_contract import embedding_model_from_environment


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
PAPER_PROTOCOL = ROOT / "official_sources/PiEvo/assets/2602.06448v1.pdf"
PAPER_PROVENANCE = ROOT / "experiments/results/paper_task_provenance.json"
READINESS = ROOT / "experiments/results/official_four_task_readiness_v8.json"
ASSET_PROBE = ROOT / "experiments/results/pievo_official_asset_probe_v3.json"
RUNNER = ROOT / "experiments/run_matched_qwen235b_task.sh"
MATCHED_REPO = ROOT / "official_sources/PiEvo_matched_qwen235b"
RADP_REPO = ROOT / "official_sources/PiEvo_radp_qwen235b"
CONFIG_ROOT = ROOT / "experiments/configs"
RESULTS = ROOT / "experiments/results"

TASK_CONFIGS = {
    "NHO": ("official_pievo_nho_task.yaml", "official_pievo_nho_model.yaml"),
    "MBO": ("official_pievo_mbo_task.yaml", "official_pievo_mbo_model_short_smiles.yaml"),
    "SPO": ("official_pievo_spo_task.yaml", "official_pievo_spo_model.yaml"),
    "TMC": ("official_pievo_tmc_task.yaml", "official_pievo_tmc_model.yaml"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, status: str, details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--readiness", type=Path, default=READINESS)
    parser.add_argument("--version", default="v8")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "official_formal_preflight_afe_v8.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    readiness_path = args.readiness.resolve()
    output_path = args.output.resolve()

    if protocol_path.is_file():
        protocol = load_json(protocol_path)
        protocol_digest = sha256(protocol_path)
        checks.append(check("protocol_file", "pass", {"sha256": protocol_digest}))
    else:
        protocol = {}
        protocol_digest = None
        checks.append(check("protocol_file", "fail", str(PROTOCOL)))
        blockers.append("protocol file missing")

    paper_protocol_ok = PAPER_PROTOCOL.is_file() and protocol.get(
        "paper_protocol_source"
    ) == "official_sources/PiEvo/assets/2602.06448v1.pdf"
    checks.append(
        check(
            "paper_protocol_provenance",
            "pass" if paper_protocol_ok else "fail",
            {"path": str(PAPER_PROTOCOL), "exists": PAPER_PROTOCOL.is_file()},
        )
    )
    if not paper_protocol_ok:
        blockers.append("official repetition provenance PDF missing or not recorded")

    paper_provenance = load_json(PAPER_PROVENANCE) if PAPER_PROVENANCE.is_file() else {}
    paper_reconstruction_ok = paper_provenance.get("paper_faithful_reconstruction_ready") is True
    checks.append(
        check(
            "paper_task_definition_provenance",
            "pass" if paper_reconstruction_ok else "warning",
            {
                "artifact": str(PAPER_PROVENANCE),
                "artifact_present": PAPER_PROVENANCE.is_file(),
                "paper_faithful_reconstruction_ready": paper_reconstruction_ok,
                "official_formal_assets_ready": paper_provenance.get("official_formal_assets_ready", False),
            },
        )
    )

    repetition = protocol.get("official_repetition_policy", {})
    tasks = repetition.get("pievo_tasks", [])
    independent_runs = repetition.get("pievo_independent_runs")
    primary_budget = protocol.get("primary_budget")
    expected_primary = 4 * 2 * 3 * 24
    matrix_ok = (
        tasks == ["NHO", "MBO", "SPO", "TMC"]
        and independent_runs == 3
        and primary_budget == 24
        and repetition.get("formal_primary_total") == expected_primary
    )
    checks.append(
        check(
            "formal_matrix",
            "pass" if matrix_ok else "fail",
            {
                "tasks": tasks,
                "independent_runs_per_task_method": independent_runs,
                "primary_budget": primary_budget,
                "expected_primary_total": expected_primary,
                "protocol_primary_total": repetition.get("formal_primary_total"),
            },
        )
    )
    if not matrix_ok:
        blockers.append("protocol formal matrix does not match 4 x 2 x 3 x 24")

    statistical = protocol.get("statistical_policy", {})
    statistical_ok = (
        statistical.get("primary_endpoint") == "mean_task_normalized_sq"
        and statistical.get("paired_unit") == "task_seed"
        and statistical.get("paired_cell_count") == 12
        and statistical.get("bootstrap", {}).get("interval") == "percentile_95"
        and statistical.get("permutation") == "exact_paired_sign_permutation_two_sided"
        and statistical.get("no_post_hoc_endpoint_or_cell_deletion") is True
    )
    checks.append(
        check(
            "paired_statistics_policy",
            "pass" if statistical_ok else "fail",
            {
                "primary_endpoint": statistical.get("primary_endpoint"),
                "paired_unit": statistical.get("paired_unit"),
                "paired_cell_count": statistical.get("paired_cell_count"),
                "bootstrap_interval": statistical.get("bootstrap", {}).get("interval"),
                "permutation": statistical.get("permutation"),
                "no_post_hoc_endpoint_or_cell_deletion": statistical.get("no_post_hoc_endpoint_or_cell_deletion"),
            },
        )
    )
    if not statistical_ok:
        blockers.append("paired strict-effect statistics policy is missing or inconsistent")

    staged = protocol.get("staged_execution", {})
    staged_ok = (
        staged.get("evaluator_g0_must_precede_matched_formal") is True
        and staged.get("official_phase_must_precede_external_full") is True
        and staged.get("external_protocol_frozen_before_official_gate") is True
        and staged.get("external_full_run_requires_gate_artifact") is True
        and staged.get("gate_failure_interpretation") == "external_not_evaluated_due_to_preregistered_stop_rule"
    )
    checks.append(check("staged_execution_policy", "pass" if staged_ok else "fail", staged))
    if not staged_ok:
        blockers.append("official-first staged execution policy is missing or inconsistent")

    execution_safety = protocol.get("execution_safety", {})
    execution_safety_ok = (
        execution_safety.get("formal_runner_requires_g0_preflight") is True
        and execution_safety.get("matched_formal_requires_evaluator_g0") is True
        and execution_safety.get("formal_run_track") == "matched_qwen235b"
        and execution_safety.get("reconstruction_track") == "matched_qwen235b_reconstruction"
        and execution_safety.get("reconstruction_can_never_enter_official_matrix") is True
        and execution_safety.get("external_runner_requires_gate_decision_and_g0") is True
    )
    checks.append(
        check(
            "execution_safety",
            "pass" if execution_safety_ok else "fail",
            execution_safety,
        )
    )
    if not execution_safety_ok:
        blockers.append("formal runner fail-closed execution safety is missing or inconsistent")

    runner_syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True, check=False
    )
    checks.append(
        check(
            "runner_syntax",
            "pass" if runner_syntax.returncode == 0 else "fail",
            runner_syntax.stderr.strip() or "bash -n passed",
        )
    )
    if runner_syntax.returncode != 0:
        blockers.append("matched runner failed bash -n")

    embedding_model = embedding_model_from_environment(os.environ)
    checks.append(
        check(
            "local_embedding_model",
            "pass" if embedding_model.get("passed") is True else "fail",
            embedding_model,
        )
    )
    if embedding_model.get("passed") is not True:
        blockers.append("local Qwen embedding model is missing, incomplete, or not offline")

    runner_text = RUNNER.read_text(encoding="utf-8") if RUNNER.is_file() else ""
    path_injection_ok = (
        'export PIEVO_CHEMBL_MODEL_PATH="${PIEVO_CHEMBL_MODEL_PATH:-$OFFICIAL_REPO/'
        in runner_text
        and 'PIEVO_CHEMBL_MODEL_PATH: "${PIEVO_CHEMBL_MODEL_PATH:-}"'
        in (CONFIG_ROOT / "official_pievo_mbo_task.yaml").read_text(encoding="utf-8")
    )
    checks.append(
        check(
            "method_specific_mbo_surrogate_path",
            "pass" if path_injection_ok else "fail",
            "runner exports the selected worktree path and MBO YAML expands it",
        )
    )
    if not path_injection_ok:
        blockers.append("MBO surrogate path is not method-specific")

    required_files = {
        "runner": RUNNER,
        "matched_main": MATCHED_REPO / "pievo/main.py",
        "radp_main": RADP_REPO / "pievo/main.py",
        "matched_adapter": MATCHED_REPO / "pievo/tools/tool_scientific_official.py",
        "radp_adapter": RADP_REPO / "pievo/tools/tool_scientific_official.py",
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    checks.append(
        check(
            "runner_and_worktrees",
            "pass" if not missing else "fail",
            {"missing": missing, "runner_executable": os.access(RUNNER, os.X_OK)},
        )
    )
    if missing or not os.access(RUNNER, os.X_OK):
        blockers.append("runner or method worktree files missing/not executable")

    yaml_status: dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        for task, (task_name, model_name) in TASK_CONFIGS.items():
            task_path = CONFIG_ROOT / task_name
            model_path = CONFIG_ROOT / model_name
            task_cfg = yaml.safe_load(task_path.read_text(encoding="utf-8")) if task_path.is_file() else None
            model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8")) if model_path.is_file() else None
            yaml_status[task] = {
                "task_file": task_path.is_file(),
                "model_file": model_path.is_file(),
                "task_yaml_mapping": isinstance(task_cfg, dict),
                "model_yaml_mapping": isinstance(model_cfg, dict),
            }
    except Exception as exc:  # pragma: no cover - environment-specific import
        yaml_status = {"error": f"{type(exc).__name__}: {exc}"}
    yaml_ok = all(
        value.get("task_file")
        and value.get("model_file")
        and value.get("task_yaml_mapping")
        and value.get("model_yaml_mapping")
        for value in yaml_status.values()
        if isinstance(value, dict) and "task_file" in value
    ) and len(yaml_status) == len(TASK_CONFIGS)
    checks.append(check("task_and_model_configs", "pass" if yaml_ok else "fail", yaml_status))
    if not yaml_ok:
        blockers.append("one or more task/model YAML files are missing or invalid")

    dependency_names = [
        "yaml",
        "torch",
        "rdkit",
        "pandas",
        "scipy",
        "sklearn",
        "flask",
        "autogen_agentchat",
        "autogen_core",
        "autogen_ext",
        "openai",
    ]
    dependencies = {
        name: bool(importlib.util.find_spec(name)) for name in dependency_names
    }
    deps_ok = all(dependencies.values())
    checks.append(check("runner_dependencies", "pass" if deps_ok else "fail", dependencies))
    if not deps_ok:
        blockers.append("runner dependency missing in project .venv")

    readiness: dict[str, Any] = {}
    formal_exact_ready = False
    formal_matched_ready = False
    reconstruction_ready = False
    if readiness_path.is_file():
        readiness = load_json(readiness_path)
        formal_exact_ready = readiness.get("formal_official_run_ready") is True
        formal_matched_ready = readiness.get("formal_matched_run_ready") is True
        reconstruction_ready = readiness.get("matched_reconstruction", {}).get("ready") is True
        checks.append(
            check(
                "matched_evaluator_assets",
                "pass" if formal_matched_ready else "blocked",
                {
                    "formal_matched_run_ready": formal_matched_ready,
                    "matched_evaluator_assets_ready": readiness.get("matched_evaluator_assets", {}).get("ready"),
                    "matched_reconstruction_ready": reconstruction_ready,
                    "task_boundaries": {
                        task: readiness.get("tasks", {}).get(task, {}).get("boundary")
                        for task in TASK_CONFIGS
                    },
                },
            )
        )
        checks.append(
            check(
                "published_official_assets",
                "pass" if formal_exact_ready else "warning",
                {
                    "formal_official_run_ready": formal_exact_ready,
                    "claim_boundary": "exact paper leaderboard claim remains locked" if not formal_exact_ready else "exact official assets verified",
                },
            )
        )
    else:
        checks.append(check("matched_evaluator_assets", "fail", "readiness file missing"))
        checks.append(check("published_official_assets", "warning", "readiness file missing"))
        blockers.append("readiness audit missing")
    if not formal_matched_ready:
        blockers.append("released evaluator/candidate-constraint assets are not ready for matched formal runs")

    asset_probe = load_json(ASSET_PROBE) if ASSET_PROBE.is_file() else {}
    probe_consistent = (
        asset_probe.get("formal_official_assets_verified") is False
        and asset_probe.get("matched_reconstruction_assets_verified") is True
        and asset_probe.get("decision") == "G0_BLOCKED_OFFICIAL_TASK_ASSETS_NOT_PUBLICLY_AVAILABLE"
    )
    checks.append(
        check(
            "upstream_asset_probe",
            "blocked" if probe_consistent else ("warning" if not asset_probe else "fail"),
            {
                "artifact": str(ASSET_PROBE),
                "artifact_present": ASSET_PROBE.is_file(),
                "decision": asset_probe.get("decision"),
                "formal_official_assets_verified": asset_probe.get("formal_official_assets_verified"),
            },
        )
    )
    if not probe_consistent and not readiness_path.is_file():
        blockers.append("upstream official asset probe is missing or inconsistent")

    exact_apd = readiness.get("exact_apd_evaluator_present") is True
    checks.append(
        check(
            "exact_apd_evaluator",
            "pass" if exact_apd else "warning",
            "APD may not be claimed until the task-specific evaluator is present",
        )
    )

    output = {
        "schema_version": f"official_formal_preflight.{args.version}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_track": protocol.get("model_track", "matched_qwen235b_only"),
        "embedding_model": embedding_model,
        "model": protocol.get("model"),
        "protocol_sha256": protocol_digest,
        "readiness_path": str(readiness_path),
        "formal_primary_total": expected_primary,
        "gate_levels": protocol.get("gate_levels", {}),
        "checks": checks,
        # APD is a claim-level prerequisite, not a prerequisite for running
        # the clean/SQ/AUOC matrix when APD is explicitly excluded.
        "formal_matched_run_ready": bool(not blockers and formal_matched_ready),
        "formal_official_run_ready": bool(not blockers and formal_exact_ready),
        "matched_reconstruction_ready": bool(reconstruction_ready),
        "upstream_asset_probe": asset_probe,
        "external_full_run_unlocked": False,
        "blockers": blockers,
        "decision": (
            "READY_FOR_MATCHED_FORMAL"
            if not blockers and formal_matched_ready
            else "BLOCKED_UNTIL_MATCHED_EVALUATOR_ASSETS_ARE_VERIFIED"
        ),
        "claim_boundary": {
            "surrogate_smoke": "development_only",
            "matched_reconstruction": "not_official_leaderboard_reproduction",
            "matched_formal": "released-evaluator Qwen3-235B comparison; not exact paper Qwen3-32B leaderboard reproduction",
            "exact_paper_setting": "locked until separate GP exact paper-setting gate",
            "external_benchmark": "locked_until_G0_G1_G2_clean_G2_reliability_G2_statistics_pass",
            "apd": "non_claimable_without_exact_evaluator",
        },
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "decision": output["decision"],
        "formal_matched_run_ready": output["formal_matched_run_ready"],
        "formal_official_run_ready": output["formal_official_run_ready"],
        "matched_reconstruction_ready": output["matched_reconstruction_ready"],
        "blocker_count": len(blockers),
    }, ensure_ascii=False, indent=2))
    return 0 if output["formal_matched_run_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
