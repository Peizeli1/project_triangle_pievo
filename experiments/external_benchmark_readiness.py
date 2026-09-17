#!/usr/bin/env python3
"""Offline readiness audit for the post-gate external benchmarks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "official_sources"
RESULTS = ROOT / "experiments/results"
NEWTON = SOURCES / "NewtonBench"
BOXING = SOURCES / "BoxingGym"
CAUSAL = SOURCES / "CausalGame"
OUTPUT = RESULTS / "external_benchmark_readiness.json"
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
PREFLIGHT = RESULTS / "official_formal_preflight_afe_v8.json"
GATE_DECISION = RESULTS / "sota_gate_v8.json"


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def importable(names: list[str]) -> dict[str, bool]:
    return {name: bool(importlib.util.find_spec(name)) for name in names}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def scenario_dirs(repo: Path) -> list[str]:
    root = repo / "experiments"
    return sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []


def main() -> int:
    protocol = load_json(PROTOCOL)
    protocol_repetitions = (
        protocol.get("official_repetition_policy", {}).get("external_repetitions", {})
        if isinstance(protocol.get("official_repetition_policy", {}), dict)
        else {}
    )
    expected_repetitions = {"NewtonBench": 4, "BoxingGym": 5, "CausalGame": 3}
    repetitions_match = protocol_repetitions == expected_repetitions
    staged = protocol.get("staged_execution", {})
    staged_execution_consistent = (
        staged.get("official_phase_must_precede_external_full") is True
        and staged.get("external_protocol_frozen_before_official_gate") is True
        and staged.get("external_full_run_requires_gate_artifact") is True
        and staged.get("gate_failure_interpretation") == "external_not_evaluated_due_to_preregistered_stop_rule"
    )
    repetition_provenance = {
        name: protocol.get("external_source_provenance", {}).get(name, {})
        for name in expected_repetitions
    }
    newton_modules = sorted(
        path.name for path in (NEWTON / "modules").iterdir()
        if path.is_dir() and path.name.startswith("m") and path.name[1:2].isdigit()
    ) if (NEWTON / "modules").is_dir() else []
    boxing_envs = sorted(
        path.stem for path in (BOXING / "src/boxing_gym/envs").glob("*.py")
        if path.name != "__init__.py" and path.stem != "goal"
    ) if (BOXING / "src/boxing_gym/envs").is_dir() else []
    causal_scenarios = scenario_dirs(CAUSAL)

    newton = {
        "source_present": NEWTON.is_dir(),
        "source_commit": git_head(NEWTON),
        "module_count": len(newton_modules),
        "module_names": newton_modules,
        "expected_tasks": 324,
        "task_count_formula": "12 domains x 3 difficulties x 3 law versions x 3 systems",
        "task_count_verified": len(newton_modules) == 12,
        "runner_present": (NEWTON / "run_master.py").is_file(),
        "evaluator_present": (NEWTON / "modules/common/evaluation.py").is_file(),
        "requirements_present": (NEWTON / "requirements.txt").is_file(),
        "dependencies": importable(["numpy", "scipy", "sympy", "openai", "dotenv"]),
        "formal_full_ready": False,
        "boundary": "Official source and evaluator are present; Qwen adapter, matched baseline, and the four-run per-method matrix are not yet executed.",
    }
    boxing = {
        "source_present": BOXING.is_dir(),
        "source_commit": git_head(BOXING),
        "environment_count": len(boxing_envs),
        "environment_names": boxing_envs,
        "expected_environments": 10,
        "environment_count_verified": len(boxing_envs) == 10,
        "runner_present": (BOXING / "run_eig_regret.py").is_file() and (BOXING / "run_experiment.py").is_file(),
        "evaluator_present": len(boxing_envs) == 10 and (BOXING / "run_eig_regret.py").is_file(),
        "requirements_present": (BOXING / "requirements.txt").is_file(),
        "dependencies": importable(["numpy", "scipy", "pandas", "pymc", "arviz", "hydra", "omegaconf", "openai"]),
        "formal_full_ready": False,
        "boundary": "Official source and environment configs are present; required probabilistic dependencies, Qwen adapter, and the five-run per-method matrix are not yet completed.",
    }
    causal_pilot = RESULTS / "causalgame_antenna_trap_seed0/summary.json"
    causal = {
        "source_present": CAUSAL.is_dir(),
        "source_commit": git_head(CAUSAL),
        "scenario_count": len(causal_scenarios),
        "scenario_names": causal_scenarios,
        "expected_scenarios": 14,
        "scenario_count_verified": len(causal_scenarios) == 14,
        "runner_present": (CAUSAL / "run_agent.py").is_file(),
        "evaluator_present": (CAUSAL / "api/modules/game/judge.py").is_file(),
        "pilot_summary_present": causal_pilot.is_file(),
        "pilot_summary_sha256": sha256(causal_pilot),
        "formal_full_ready": False,
        "boundary": "Official 14-scenario source is present; retained evidence is one antenna_trap policy replay with no LLM calls, not the required three-run interactive full benchmark.",
    }
    tasks = {"NewtonBench": newton, "BoxingGym": boxing, "CausalGame": causal}
    source_scopes_verified = all(
        value.get("source_present") and value.get("runner_present") and value.get("evaluator_present")
        for value in tasks.values()
    )
    preflight = load_json(PREFLIGHT)
    gate_decision = load_json(GATE_DECISION)
    g0_ready = preflight.get("formal_matched_run_ready") is True
    exact_paper_ready = preflight.get("formal_official_run_ready") is True
    metric_gate_passed = gate_decision.get("external_full_run_unlocked") is True
    external_full_run_unlocked = bool(
        source_scopes_verified
        and repetitions_match
        and staged_execution_consistent
        and g0_ready
        and metric_gate_passed
    )
    result = {
        "schema_version": "external_benchmark_readiness.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "required_repetitions_per_method": expected_repetitions,
        "repetition_provenance": repetition_provenance,
        "protocol_repetition_consistency": repetitions_match,
        "staged_execution_consistency": staged_execution_consistent,
        "official_superiority_gate_required": True,
        "g0_official_ready": g0_ready,
        "g0_matched_evaluator_ready": g0_ready,
        "g0_exact_paper_assets_ready": exact_paper_ready,
        "official_metric_gate_passed": metric_gate_passed,
        "external_full_run_unlocked": external_full_run_unlocked,
        "benchmarks": tasks,
        "all_source_scopes_verified": source_scopes_verified and repetitions_match,
        "all_formal_full_ready": all(value.get("formal_full_ready") is True for value in tasks.values()),
        "claim_boundary": {
            "pilot": "development/transfer evidence only",
            "external_sota": "forbidden before official gate and complete repetitions",
            "partial_coverage": "not leaderboard/full benchmark claim",
            "unlock_rule": "matched evaluator G0 preflight and metric G1/G2 clean/reliability/statistics gates must all pass",
            "paper_setting_claim": "forbidden unless separate Qwen3-32B exact paper-setting gate passes",
            "gate_failure": "external_not_evaluated_due_to_preregistered_stop_rule",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "all_source_scopes_verified": result["all_source_scopes_verified"],
        "all_formal_full_ready": result["all_formal_full_ready"],
        "external_full_run_unlocked": result["external_full_run_unlocked"],
        "newton_tasks": newton["module_count"],
        "boxing_environments": boxing["environment_count"],
        "causal_scenarios": causal["scenario_count"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["all_formal_full_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
