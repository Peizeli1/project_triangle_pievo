#!/usr/bin/env python3
"""Preregistered Qwen3-235B component suite for Triage-PiEvo/RADP.

Four arms are compared under the same released evaluators, tasks, seeds and
primary budget.  This is a bounded development-track ablation, separate from
any existing main-table artifacts.

Arms:
  pievo          : released matched PiEvo path
  triage_only    : RADP evidence gate/revision lifecycle, legacy candidate search
  radp_no_gate   : current AFE/RADP stack with source-gate authorization disabled
  radp_full      : current AFE + source-gate RADP stack

"radp_no_gate" is intentionally not called "AFE-only": it removes the gate
from the implemented RADP bundle while retaining the rest of that bundle.
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

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/run_matched_qwen235b_task.sh"
BASH = Path("/opt/homebrew/bin/bash")
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v11.json"
DECODE = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"
MODEL = "qwen3-235b-a22b-instruct-2507"
MODEL_TRACK = "matched_qwen235b"
TASKS = ("NHO", "MBO", "SPO", "TMC")
SEEDS = (0, 1, 2)
BUDGET = 8
ARMS = ("pievo", "triage_only", "radp_no_gate", "radp_full")
OUTPUT = ROOT / "experiments/results/qwen_required_suite_v1/factorial"
PREFLIGHT_OUTPUT = ROOT / "experiments/results/qwen_required_suite_v1/preflight_factorial"
PLAN = OUTPUT / "factorial_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cells() -> list[str]:
    return [f"{task}:{arm}:{seed}" for task in TASKS for arm in ARMS for seed in SEEDS]


def arm_env(task: str, arm: str) -> dict[str, str]:
    if arm == "pievo":
        return {"PIEVO_METHOD": "pievo"}
    strategy = "afe_spo_v2" if task == "SPO" else "afe_v9"
    base = {
        "PIEVO_METHOD": "radp",
        "PIEVO_SOURCE_GATE_BEHAVIOR": "fixed",
    }
    if arm == "triage_only":
        base.update({
            "PIEVO_RADP_SEARCH_STRATEGY": "legacy_v1",
            "PIEVO_AFE_FORCE_ENABLED": "0",
            "PIEVO_EXPLORATORY_AFE_GATE_MODES": "0",
        })
    elif arm == "radp_no_gate":
        base.update({
            "PIEVO_RADP_SEARCH_STRATEGY": strategy,
            "PIEVO_AFE_FORCE_ENABLED": "1",
            "PIEVO_EXPLORATORY_AFE_GATE_MODES": "1",
            "PIEVO_EXPLORATORY_GATE_MODE": "off",
        })
    elif arm == "radp_full":
        base.update({
            "PIEVO_RADP_SEARCH_STRATEGY": strategy,
            "PIEVO_AFE_FORCE_ENABLED": "0",
            "PIEVO_EXPLORATORY_AFE_GATE_MODES": "0",
        })
    else:
        raise ValueError(arm)
    return base


def expected_plan() -> dict[str, Any]:
    return {
        "schema_version": "qwen_required_factorial.v1",
        "model": MODEL,
        "model_track": MODEL_TRACK,
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": BUDGET,
        "arms": list(ARMS),
        "arm_definitions": {
            "pievo": "released matched PiEvo candidate search and principle update path",
            "triage_only": "source-aware gate/revision lifecycle with legacy candidate search and AFE disabled",
            "radp_no_gate": "implemented RADP bundle with AFE retained and source-gate authorization disabled",
            "radp_full": "implemented RADP bundle with AFE and active source-gate authorization",
        },
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "decode_path": str(DECODE.resolve()),
        "decode_sha256": sha256(DECODE),
        "runner_sha256": sha256(RUNNER),
        "selection_rule": "retain all predeclared cells; no outcome-dependent retries, deletion, seed selection, or condition reordering",
        "claim_boundary": "bounded Qwen component ablation; does not replace the frozen main-table protocol and must be reported with its 8-primary budget",
        "selected_cells": cells(),
    }


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = expected_plan()
    if not PLAN.exists():
        write_json(PLAN, expected)
    observed = json.loads(PLAN.read_text(encoding="utf-8"))
    if observed != expected:
        raise SystemExit(f"preregistered plan mismatch: {PLAN}")


def run_cell(spec: str, dry_run: bool) -> dict[str, Any]:
    task, arm, seed_text = spec.split(":")
    seed = int(seed_text)
    root = PREFLIGHT_OUTPUT if dry_run else OUTPUT
    out = root / f"qwen235b_{task.lower()}_{arm}_seed{seed}_budget{BUDGET}"
    if out.exists() and any(out.iterdir()):
        return {"cell": spec, "status": "refused_nonfresh_output", "returncode": 3, "output": str(out)}
    env = os.environ.copy()
    env.pop("ZJU_TASK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update({
        "PIEVO_TASK": task,
        "PIEVO_SEED": str(seed),
        "PIEVO_BUDGET": str(BUDGET),
        "PIEVO_RUN_STAGE": "development",
        "PIEVO_RUN_TRACK": f"{MODEL_TRACK}_reconstruction",
        "PIEVO_MODEL_TRACK": MODEL_TRACK,
        "PIEVO_AGENT_MODEL": MODEL,
        "PIEVO_PROTOCOL_PATH": str(PROTOCOL.resolve()),
        "PIEVO_DECODE_PROTOCOL": str(DECODE.resolve()),
        "PIEVO_OUT": str(out.resolve()),
        "PIEVO_MAX_TOTAL_TOKENS": "100000",
        "PIEVO_AGENT_MAX_TOKENS": "4096",
        "PIEVO_MAX_RETRIES": "0",
        "PIEVO_CORRUPTION_MODE": "none",
        "PIEVO_PAPER_ELIGIBLE": "0",
    })
    env.update(arm_env(task, arm))
    cmd = [str(BASH), str(RUNNER)]
    process = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**env, "PIEVO_DRY_RUN": "1" if dry_run else "0"},
        capture_output=True,
        text=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / ("preflight.log" if dry_run else "run.log")
    log_path.write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8")
    return {
        "cell": spec,
        "status": "passed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "output": str(out.resolve()),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    if not BASH.is_file() or not RUNNER.is_file() or not PROTOCOL.is_file() or not DECODE.is_file():
        raise SystemExit("missing Qwen suite dependency")
    prepare()
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "cells": len(cells()), "plan": str(PLAN)}, indent=2))
        return 0
    execution_path = (PREFLIGHT_OUTPUT if args.dry_run else OUTPUT) / "execution_manifest.json"
    if execution_path.exists():
        raise SystemExit(f"execution already exists: {execution_path}")
    write_json(execution_path, {
        "schema_version": "qwen_required_factorial_execution.v1",
        "plan_sha256": sha256(PLAN),
        "dry_run": args.dry_run,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_cells": cells(),
        "status": "running",
    })
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(run_cell, spec, args.dry_run): spec for spec in cells()}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda r: r["cell"])
    write_json(execution_path.parent / "cell_status.json", results)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution.update({
        "status": "completed" if all(r["status"] == "passed" for r in results) else "completed_with_failures",
        "passed_cells": sum(r["status"] == "passed" for r in results),
        "failed_cells": sum(r["status"] != "passed" for r in results),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(execution_path, execution)
    print(json.dumps({"summary": execution}, indent=2))
    return 0 if execution["failed_cells"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
