#!/usr/bin/env python3
"""Freeze the v8 matched AFE protocol before any formal API collection.

The artifact records hashes and source commits only.  It never contacts an
LLM, changes a result directory, or stores credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from formal_runtime_contract import embedding_model_from_environment


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
READINESS = ROOT / "experiments/results/official_four_task_readiness_v8.json"
PREFLIGHT = ROOT / "experiments/results/official_formal_preflight_afe_v8.json"
OUTPUT = ROOT / "experiments/results/matched_qwen235b_protocol_freeze_afe_v8.json"
G0 = ROOT / "experiments/results/g0_evidence_audit_v8.json"
G1 = ROOT / "experiments/results/g1_fairness_audit_v8.json"

TASK_CONFIGS = [
    ROOT / "experiments/configs/official_pievo_nho_task.yaml",
    ROOT / "experiments/configs/official_pievo_nho_model.yaml",
    ROOT / "experiments/configs/official_pievo_mbo_task.yaml",
    ROOT / "experiments/configs/official_pievo_mbo_model_short_smiles.yaml",
    ROOT / "experiments/configs/official_pievo_spo_task.yaml",
    ROOT / "experiments/configs/official_pievo_spo_model.yaml",
    ROOT / "experiments/configs/official_pievo_tmc_task.yaml",
    ROOT / "experiments/configs/official_pievo_tmc_model.yaml",
]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--readiness", type=Path, default=READINESS)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--g0", type=Path, default=G0)
    parser.add_argument("--g1", type=Path, default=G1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    readiness_path = args.readiness.resolve()
    preflight_path = args.preflight.resolve()
    g0_path = args.g0.resolve()
    g1_path = args.g1.resolve()
    output_path = args.output.resolve()
    protocol = load_json(protocol_path)
    readiness = load_json(readiness_path)
    preflight = load_json(preflight_path)
    g0 = load_json(g0_path)
    g1 = load_json(g1_path)
    external = protocol.get("external_source_provenance", {})
    source_commits = {
        name: {
            "local_path": value.get("local_path"),
            "declared_commit": value.get("commit"),
            "observed_commit": git_head(ROOT / str(value.get("local_path", ""))),
            "repetition_count": value.get("repetition_count"),
            "repetition_semantics": value.get("repetition_semantics"),
        }
        for name, value in external.items()
        if isinstance(value, dict)
    }
    embedding_model = embedding_model_from_environment(os.environ)
    result = {
        "schema_version": "matched_qwen235b_protocol_freeze.v2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model": protocol.get("model"),
            "model_track": protocol.get("model_track"),
            "formal_comparison_scope": protocol.get("official_repetition_policy", {}).get("formal_comparison_scope"),
            "tasks": protocol.get("official_repetition_policy", {}).get("pievo_tasks"),
            "methods": protocol.get("g1_fairness_protocol", {})
            .get("formal_primary_comparison", {})
            .get("arms", ["strongest_pievo", "radp_afe"]),
            "formal_seeds": protocol.get("seed_policy", {}).get("formal"),
            "primary_budget": protocol.get("primary_budget"),
            "formal_primary_total": protocol.get("official_repetition_policy", {}).get("formal_primary_total"),
        },
        "protocol_sha256": sha256(protocol_path),
        "readiness_sha256": sha256(readiness_path),
        "preflight_sha256": sha256(preflight_path),
        "g0_evidence_sha256": sha256(g0_path),
        "g1_fairness_sha256": sha256(g1_path),
        "g0_state": {"passed": g0.get("passed"), "schema_version": g0.get("schema_version")},
        "g1_state": {"passed": g1.get("passed"), "schema_version": g1.get("schema_version")},
        "readiness_state": {
            "formal_matched_run_ready": readiness.get("formal_matched_run_ready"),
            "formal_official_run_ready": readiness.get("formal_official_run_ready"),
            "exact_apd_evaluator_present": readiness.get("exact_apd_evaluator_present"),
            "matched_evaluator_assets_ready": readiness.get("matched_evaluator_assets", {}).get("ready"),
        },
        "preflight_state": {
            "decision": preflight.get("decision"),
            "formal_matched_run_ready": preflight.get("formal_matched_run_ready"),
            "formal_official_run_ready": preflight.get("formal_official_run_ready"),
            "blockers": preflight.get("blockers", []),
        },
        "task_config_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in TASK_CONFIGS
        },
        "runner_sha256": sha256(ROOT / "experiments/run_matched_qwen235b_task.sh"),
        "embedding_model": embedding_model,
        "metric_script_sha256": sha256(ROOT / "experiments/official_four_task_metrics.py"),
        "worktree_core_hashes": {
            "pievo_main": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/main.py"),
            "radp_main": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/main.py"),
            "pievo_evolveflow": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/group/evolveflow.py"),
            "radp_evolveflow": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/group/evolveflow.py"),
            "pievo_manage": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/group/manage.py"),
            "radp_manage": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/group/manage.py"),
            "pievo_experiment_agent": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/agents/experiment.py"),
            "radp_experiment_agent": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/agents/experiment.py"),
            "pievo_instruction": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/word/instruction.py"),
            "radp_instruction": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/word/instruction.py"),
            "pievo_embedding": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/word/embedding.py"),
            "radp_embedding": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/word/embedding.py"),
            "pievo_candidate_validation": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/tools/candidate_validation.py"),
            "radp_candidate_validation": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/tools/candidate_validation.py"),
            "pievo_chembl_tool": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/tools/tool_chembl_official.py"),
            "radp_chembl_tool": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/tools/tool_chembl_official.py"),
            "pievo_model_config": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/config/model.yaml"),
            "radp_model_config": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/config/model.yaml"),
            "pievo_bayesian": sha256(ROOT / "official_sources/PiEvo_matched_qwen235b/pievo/word/bayesian.py"),
            "radp_bayesian": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/word/bayesian.py"),
            "radp_afe": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/radp/afe.py"),
            "radp_wide_candidate": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/radp/wide_candidate.py"),
            "radp_capabilities": sha256(ROOT / "official_sources/PiEvo_radp_qwen235b/pievo/radp/capabilities.json"),
            "common_candidate_generator": sha256(ROOT / "experiments/common_candidate_generator.py"),
        },
        "source_commits": source_commits,
        "claim_boundary": {
            "matched_claim": "permitted only after G0/G1/G2 matched gates",
            "exact_paper_claim": "forbidden without separate Qwen3-32B exact paper-setting gate",
            "external_full": "forbidden until gate artifact is true",
            "apd": "non-claimable without exact task-specific evaluator",
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "protocol_sha256": result["protocol_sha256"],
        "formal_matched_run_ready": result["preflight_state"]["formal_matched_run_ready"],
        "formal_official_run_ready": result["preflight_state"]["formal_official_run_ready"],
        "external_full_forbidden_until_gate": True,
    }, ensure_ascii=False, indent=2))
    return 0 if (
        result["preflight_state"]["formal_matched_run_ready"] is True
        and g0.get("passed") is True
        and g1.get("passed") is True
        and embedding_model.get("passed") is True
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
