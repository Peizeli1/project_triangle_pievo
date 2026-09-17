#!/usr/bin/env python3
"""Run the preregistered four-condition AFE/gate confound audit.

This is a paper-ineligible, development-track experiment.  It keeps the
model, evaluator, budget, seed set, and injected anomaly schedule fixed while
varying the implementation factors needed to separate the released PiEvo
path, the legacy AFE path, the gate behavior, and the current AFE path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_matched_qwen235b_task.sh"
PROTOCOL = ROOT / "experiments/configs/gpt56luna_medium_pievo_radp_protocol_v1.json"
OUTPUT = ROOT / "experiments/results/exploratory_afe_gate_modes_gpt56luna_20260813_v5"
PLAN = OUTPUT / "afe_gate_modes_protocol.json"
SEEDS = (0, 1)
BUDGET = 6

CONDITIONS: dict[str, dict[str, str]] = {
    "afe_v8_off": {
        "method": "radp",
        "afe": "afe_v8",
        "gate": "off",
        "strategy": "afe_v8",
        "behavior": "off",
    },
    "afe_v8_shadow": {
        "method": "radp",
        "afe": "afe_v8",
        "gate": "shadow",
        "strategy": "afe_v8",
        "behavior": "shadow",
    },
    "afe_v8_active": {
        "method": "radp",
        "afe": "afe_v8",
        "gate": "active",
        "strategy": "afe_v8",
        "behavior": "fixed",
    },
    "afe_v9_active": {
        "method": "radp",
        "afe": "afe_v9",
        "gate": "active",
        "strategy": "afe_v9",
        "behavior": "fixed",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cells() -> list[str]:
    return [f"{condition}:{seed}" for condition in CONDITIONS for seed in SEEDS]


def expected_plan() -> dict[str, Any]:
    return {
        "schema_version": "gpt56luna_afe_gate_modes.v1",
        "paper_eligible": False,
        "include_in_main_paper_tables": False,
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "task": "MBO",
        "seeds": list(SEEDS),
        "primary_budget_per_cell": BUDGET,
        "conditions": CONDITIONS,
        "selected_cells": cells(),
        "fixed_controls": {
            "protocol": str(PROTOCOL.resolve()),
            "protocol_sha256": sha256(PROTOCOL),
            "corruption_mode": "one_shot_spike",
            "corruption_calls": [2, 3],
            "corruption_offset": 2.0,
            "evaluator": "released_local_chembl35_surrogate",
            "same_model_evaluator_budget_and_seed_set": True,
        },
        "factor_interpretation": {
            "afe_v8_off": "legacy RADP AFE v8 with the source gate disabled",
            "afe_v8_shadow": "legacy RADP AFE v8 with source diagnostics logged but update authorization unchanged",
            "afe_v8_active": "legacy RADP AFE v8 with source-gate update authorization and repair enabled",
            "afe_v9_active": "current RADP AFE v9 with source-gate update authorization and repair enabled",
        },
        "selection_rule": "retain every predeclared cell; no score-based retries, deletion, or condition reordering",
        "claim_boundary": "This isolates implementation factors under a controlled released surrogate; it does not establish real-world diagnosis, independent chemistry verification, or universal SOTA.",
    }


def validate_plan() -> dict[str, Any]:
    if not PLAN.is_file():
        raise SystemExit(f"missing preregistered plan: {PLAN}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    expected = expected_plan()
    for key, value in expected.items():
        if plan.get(key) != value:
            raise SystemExit(f"plan mismatch for {key}: {plan.get(key)!r}")
    return plan


def run_cell(spec: str) -> dict[str, Any]:
    condition, seed_text = spec.split(":")
    seed = int(seed_text)
    definition = CONDITIONS[condition]
    method = definition["method"]
    out = OUTPUT / f"gpt56luna_{condition}_mbo_seed{seed}_budget{BUDGET}"
    if out.exists() and any(out.iterdir()):
        # A preflight-only failure can be retried in a new explicitly versioned
        # output root.  Completed cells in any prior root remain immutable.
        # completed cells remain immutable and their artifacts are retained.
        if (out / "run_manifest.json").is_file() or (out / "official_chembl_tool_audit.jsonl").is_file():
            return {"cell": spec, "status": "refused_nonfresh_output", "returncode": 3, "output": str(out)}
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{condition}_seed{seed}.log"
    if log_path.exists():
        log_path = log_dir / f"{condition}_seed{seed}_retry1.log"
    env = os.environ.copy()
    env.pop("ZJU_TASK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update(
        {
            "PIEVO_TASK": "MBO",
            "PIEVO_METHOD": method,
            "PIEVO_SEED": str(seed),
            "PIEVO_BUDGET": str(BUDGET),
            "PIEVO_RUN_STAGE": "development",
            "PIEVO_RUN_TRACK": "gpt56luna_reconstruction",
            "PIEVO_MODEL_TRACK": "gpt56luna",
            "PIEVO_AGENT_MODEL": "gpt-5.6-luna",
            "PIEVO_REASONING_EFFORT": "medium",
            "PIEVO_PROTOCOL_PATH": str(PROTOCOL.resolve()),
            "PIEVO_DECODE_PROTOCOL": str(PROTOCOL.resolve()),
            "PIEVO_OUT": str(out.resolve()),
            "PIEVO_MAX_TOTAL_TOKENS": "100000",
            "PIEVO_AGENT_MAX_TOKENS": "4096",
            "PIEVO_PREFILTER_INVALID_CANDIDATES": "1",
            "PIEVO_MAX_PREFILTER_REJECTIONS": "12",
            "PIEVO_MAX_PREFILTER_EMPTY_POOLS": "2",
            "PIEVO_RADP_SEARCH_STRATEGY": definition["strategy"],
            "PIEVO_AFE_SCHEMA_VERSION": "v9" if definition["strategy"] == "afe_v9" else "v8",
            "PIEVO_RADP_WIDE_SEED_BANK": "0",
            "PIEVO_COMMON_GENERATOR": "0",
            "PIEVO_CORRUPTION_MODE": "one_shot_spike",
            "PIEVO_CORRUPTION_CALLS": "2,3",
            "PIEVO_CORRUPTION_OFFSET": "2.0",
            "PIEVO_SOURCE_GATE_SCORE_GE": "0.0",
            "PIEVO_SOURCE_GATE_BEHAVIOR": definition["behavior"],
            "PIEVO_EXPLORATORY_AFE_GATE_MODES": "1",
            "PIEVO_EXPLORATORY_GATE_MODE": definition["gate"],
            "PIEVO_AFE_FORCE_ENABLED": "1",
            "PIEVO_PAPER_ELIGIBLE": "0",
        }
    )
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            ["env", "PIEVO_DRY_RUN=0", "bash", str(RUNNER)],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    manifest = {}
    manifest_path = out / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    write_json(
        out / "afe_gate_modes_cell_attestation.json",
        {
            "schema_version": "gpt56luna_afe_gate_modes_cell.v1",
            "cell": spec,
            "condition": condition,
            "definition": definition,
            "plan_path": str(PLAN.resolve()),
            "plan_sha256": sha256(PLAN),
            "paper_eligible": False,
            "runner_returncode": process.returncode,
            "stdout_log": str(log_path.resolve()),
            "manifest_status": manifest.get("status"),
        },
    )
    return {
        "cell": spec,
        "condition": condition,
        "seed": seed,
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "output": str(out.resolve()),
        "manifest_status": manifest.get("status"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel must be positive")
    if not RUNNER.is_file() or not PROTOCOL.is_file():
        raise SystemExit("missing runner or protocol")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PLAN.exists():
        write_json(PLAN, expected_plan())
    validate_plan()
    if args.prepare_only:
        print(json.dumps({"event": "afe_gate_modes_prepared", "plan": str(PLAN.resolve()), "cells": len(cells())}, ensure_ascii=False))
        return 0
    execution_path = OUTPUT / "execution_manifest.json"
    if execution_path.exists():
        raise SystemExit(f"experiment already started; refusing a second run: {execution_path}")
    write_json(
        execution_path,
        {
            "schema_version": "gpt56luna_afe_gate_modes_execution.v1",
            "plan_path": str(PLAN.resolve()),
            "plan_sha256": sha256(PLAN),
            "paper_eligible": False,
            "selected_cells": cells(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        },
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_cell, spec): spec for spec in cells()}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "afe_gate_modes_cell_complete", **result}, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: item["cell"])
    write_json(OUTPUT / "cell_status.json", results)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution.update(
        {
            "status": "completed" if all(item["status"] == "completed" for item in results) else "completed_with_failures",
            "completed_cells": sum(item["status"] == "completed" for item in results),
            "failed_cells": sum(item["status"] != "completed" for item in results),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(execution_path, execution)
    return 0 if execution["failed_cells"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
