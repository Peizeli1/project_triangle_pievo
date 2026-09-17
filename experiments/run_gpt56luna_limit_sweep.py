#!/usr/bin/env python3
"""Run the pre-registered, paper-ineligible Luna seed limit sweep.

This track is intentionally separate from the formal paper artifacts.  It
uses the already frozen Luna/PiEvo-RADP protocol and evaluator, retains every
cell, and never selects seeds based on their observed scores.
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
WRAPPER = ROOT / "experiments/run_gpt56luna_pievo_radp_cell.sh"
FROZEN_PROTOCOL = ROOT / "experiments/configs/gpt56luna_medium_pievo_radp_protocol_v1.json"
OUTPUT = ROOT / "experiments/results/exploratory_limit_sweep_gpt56luna_20260812"
PLAN_PATH = OUTPUT / "limit_sweep_protocol.json"
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (3, 4, 5, 6, 7)
BUDGET = 24


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cells() -> list[str]:
    return [
        f"{method}:{task}:{seed}"
        for task in TASKS
        for method in METHODS
        for seed in SEEDS
    ]


def expected_plan() -> dict[str, Any]:
    return {
        "schema_version": "gpt56luna_exploratory_limit_sweep.v1",
        "purpose": "empirical_limit_and_variance_characterization",
        "paper_eligible": False,
        "include_in_main_paper_tables": False,
        "seed_selection": "fixed_before_run_and_all_cells_retained",
        "model": "gpt-5.6-luna",
        "model_track": "gpt56luna",
        "reasoning_effort": "medium",
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "extra_seeds": list(SEEDS),
        "primary_budget_per_cell": BUDGET,
        "selected_cells": cells(),
        "protocol_path": str(FROZEN_PROTOCOL.resolve()),
        "protocol_sha256": sha256(FROZEN_PROTOCOL),
        "evaluator": "released_local_pievo_piflow_evaluator",
        "claim_boundary": (
            "Exploratory limit characterization only; retain all completed and "
            "failed cells; do not use for paper model selection or cherry-picking."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_plan() -> dict[str, Any]:
    if not PLAN_PATH.is_file():
        raise SystemExit(f"missing pre-registered plan: {PLAN_PATH}")
    try:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid pre-registered plan: {PLAN_PATH}: {exc}") from exc
    expected = expected_plan()
    for key, value in expected.items():
        if plan.get(key) != value:
            raise SystemExit(f"pre-registered plan mismatch for {key}: {plan.get(key)!r}")
    if plan.get("paper_eligible") is not False or plan.get("include_in_main_paper_tables") is not False:
        raise SystemExit("limit sweep must be explicitly paper-ineligible")
    return plan


def run_cell(spec: str) -> dict[str, Any]:
    method, task, seed_text = spec.split(":")
    seed = int(seed_text)
    cell_output = OUTPUT / f"gpt56luna_{method}_{task.lower()}_seed{seed}_formal24"
    if cell_output.exists() and any(cell_output.iterdir()):
        return {
            "cell": spec,
            "status": "refused_nonfresh_output",
            "returncode": 3,
            "output": str(cell_output),
        }
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{method}_{task.lower()}_seed{seed}.log"
    environment = os.environ.copy()
    # Force the child to use the configured Keychain helper; never propagate a
    # raw credential from an unrelated shell environment.
    environment.pop("ZJU_TASK_API_KEY", None)
    environment.pop("OPENAI_API_KEY", None)
    environment.update(
        {
            "PIEVO_TASK": task,
            "PIEVO_METHOD": method,
            "PIEVO_SEED": str(seed),
            "PIEVO_BUDGET": str(BUDGET),
            "PIEVO_OUT": str(cell_output.resolve()),
            "PIEVO_RUN_STAGE": "formal",
            "PIEVO_RUN_TRACK": "gpt56luna",
            "PIEVO_MODEL_TRACK": "gpt56luna",
            "PIEVO_AGENT_MODEL": "gpt-5.6-luna",
            "PIEVO_REASONING_EFFORT": "medium",
            "PIEVO_PROTOCOL_PATH": str(FROZEN_PROTOCOL.resolve()),
            "PIEVO_DECODE_PROTOCOL": str(FROZEN_PROTOCOL.resolve()),
            "PIEVO_EXPLORATORY_LIMIT_SWEEP": "1",
            "PIEVO_PAPER_ELIGIBLE": "0",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            ["bash", str(WRAPPER)],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    summary_path = cell_output / "summary.json"
    manifest_path = cell_output / "run_manifest.json"
    summary: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for path, target in ((summary_path, summary), (manifest_path, manifest)):
        if path.is_file():
            try:
                target.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
    write_json(
        cell_output / "limit_sweep_cell_attestation.json",
        {
            "schema_version": "gpt56luna_exploratory_limit_sweep_cell.v1",
            "cell": spec,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "medium",
            "plan_path": str(PLAN_PATH.resolve()),
            "plan_sha256": sha256(PLAN_PATH),
            "paper_eligible": False,
            "include_in_main_paper_tables": False,
            "runner_returncode": process.returncode,
            "stdout_log": str(log_path.resolve()),
        },
    )
    return {
        "cell": spec,
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "output": str(cell_output.resolve()),
        "summary_status": summary.get("status"),
        "primary_count": summary.get("primary_count"),
        "best_output": summary.get("best_output"),
        "mean_output": summary.get("mean_output"),
        "sq_percent": summary.get("sq_percent"),
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
    if not WRAPPER.is_file() or not FROZEN_PROTOCOL.is_file():
        raise SystemExit("missing frozen Luna wrapper or protocol")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = expected_plan()
    if not PLAN_PATH.exists():
        write_json(PLAN_PATH, plan)
    validate_plan()
    if args.prepare_only:
        print(json.dumps({"event": "limit_sweep_prepared", "plan": str(PLAN_PATH.resolve()), "cells": len(cells())}, ensure_ascii=False))
        return 0
    execution_path = OUTPUT / "execution_manifest.json"
    if execution_path.exists():
        raise SystemExit(f"execution manifest already exists; refusing a second sweep: {execution_path}")
    execution = {
        "schema_version": "gpt56luna_exploratory_limit_sweep_execution.v1",
        "plan_path": str(PLAN_PATH.resolve()),
        "plan_sha256": sha256(PLAN_PATH),
        "paper_eligible": False,
        "selected_cells": cells(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    write_json(execution_path, execution)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_cell, cell): cell for cell in cells()}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "event": "limit_cell_complete",
                        "cell": result.get("cell"),
                        "status": result.get("status"),
                        "returncode": result.get("returncode"),
                        "primary": result.get("primary_count"),
                        "best": result.get("best_output"),
                        "sq_percent": result.get("sq_percent"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    results.sort(key=lambda item: item["cell"])
    write_json(OUTPUT / "cell_status.json", results)
    execution.update(
        {
            "status": "completed" if all(item.get("status") == "completed" for item in results) else "completed_with_failures",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_cells": sum(item.get("status") == "completed" for item in results),
            "failed_cells": sum(item.get("status") != "completed" for item in results),
        }
    )
    write_json(execution_path, execution)
    return 0 if execution["failed_cells"] == 0 and len(results) == len(cells()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
