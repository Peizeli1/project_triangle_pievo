#!/usr/bin/env python3
"""Run a preregistered, paper-ineligible RADP gate-behavior ablation.

The two conditions use the same RADP code path, model protocol, seed, evaluator,
budget, and injected anomaly schedule.  Only ``PIEVO_SOURCE_GATE_BEHAVIOR``
changes: ``fixed`` performs the source-gate repair/quarantine, whereas ``noop``
records the verifier disagreement but allows the update to proceed.  This is a
limit/mechanism experiment and is intentionally excluded from the manuscript.
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
OUTPUT = ROOT / "experiments/results/exploratory_gate_ablation_gpt56luna_20260812"
PLAN = OUTPUT / "gate_ablation_protocol.json"
TASK = "MBO"
METHOD = "radp"
BEHAVIORS = ("fixed", "noop")
SEEDS = (0, 1)
BUDGET = 6


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
    return [f"{METHOD}:{TASK}:{behavior}:{seed}" for behavior in BEHAVIORS for seed in SEEDS]


def expected_plan() -> dict[str, Any]:
    return {
        "schema_version": "gpt56luna_gate_behavior_ablation.v1",
        "paper_eligible": False,
        "include_in_main_paper_tables": False,
        "purpose": "isolate_the_effect_of_source_gate_update_authorization",
        "claim_boundary": (
            "Controlled evaluator-side ablation only; it does not establish real-world "
            "fault diagnosis, independent physical verification, or external SOTA."
        ),
        "model": "gpt-5.6-luna",
        "model_track": "gpt56luna",
        "reasoning_effort": "medium",
        "task": TASK,
        "method": METHOD,
        "behaviors": list(BEHAVIORS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": BUDGET,
        "selected_cells": cells(),
        "fixed_controls": {
            "corruption_mode": "one_shot_spike",
            "corruption_calls": [2, 3],
            "corruption_offset": 2.0,
            "radp_search_strategy": "afe_v8",
            "source_gate_score_ge": 0.0,
            "source_gate_values_used_for_candidate_selection": False,
            "same_schedule_across_behaviors": True,
        },
        "condition_definition": {
            "fixed": "verifier disagreement rejects the update and repairs/quarantines the internal history",
            "noop": "verifier disagreement is logged but the update remains authorized and the raw history is retained",
        },
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "evaluator": "released_local_chembl35_surrogate",
        "selection_rule": "all four cells retained; no outcome-dependent reruns or deletions",
    }


def validate_plan() -> dict[str, Any]:
    if not PLAN.is_file():
        raise SystemExit(f"missing preregistered plan: {PLAN}")
    try:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid ablation plan: {PLAN}: {exc}") from exc
    expected = expected_plan()
    for key, value in expected.items():
        if plan.get(key) != value:
            raise SystemExit(f"ablation plan mismatch for {key}: {plan.get(key)!r}")
    if plan.get("paper_eligible") is not False or plan.get("include_in_main_paper_tables") is not False:
        raise SystemExit("gate ablation must be paper-ineligible")
    return plan


def run_cell(spec: str) -> dict[str, Any]:
    method, task, behavior, seed_text = spec.split(":")
    seed = int(seed_text)
    out = OUTPUT / f"gpt56luna_{method}_{task.lower()}_seed{seed}_budget{BUDGET}_gate_{behavior}"
    if out.exists() and any(out.iterdir()):
        return {"cell": spec, "status": "refused_nonfresh_output", "returncode": 3, "output": str(out)}
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{method}_{task.lower()}_{behavior}_seed{seed}.log"
    env = os.environ.copy()
    env.pop("ZJU_TASK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update(
        {
            "PIEVO_TASK": task,
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
            "PIEVO_RADP_SEARCH_STRATEGY": "afe_v8",
            "PIEVO_AFE_SCHEMA_VERSION": "v8",
            "PIEVO_RADP_WIDE_SEED_BANK": "0",
            "PIEVO_COMMON_GENERATOR": "0",
            "PIEVO_CORRUPTION_MODE": "one_shot_spike",
            "PIEVO_CORRUPTION_CALLS": "2,3",
            "PIEVO_CORRUPTION_OFFSET": "2.0",
            "PIEVO_SOURCE_GATE_SCORE_GE": "0.0",
            "PIEVO_SOURCE_GATE_BEHAVIOR": behavior,
            "PIEVO_EXPLORATORY_GATE_ABLATION": "1",
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
        except (OSError, json.JSONDecodeError):
            pass
    write_json(
        out / "gate_ablation_cell_attestation.json",
        {
            "schema_version": "gpt56luna_gate_ablation_cell.v1",
            "cell": spec,
            "plan_path": str(PLAN.resolve()),
            "plan_sha256": sha256(PLAN),
            "paper_eligible": False,
            "gate_behavior": behavior,
            "injected_mode": "one_shot_spike",
            "injected_calls": [2, 3],
            "runner_returncode": process.returncode,
            "stdout_log": str(log_path.resolve()),
            "manifest_status": manifest.get("status"),
        },
    )
    return {
        "cell": spec,
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
        print(json.dumps({"event": "gate_ablation_prepared", "plan": str(PLAN.resolve()), "cells": len(cells())}, ensure_ascii=False))
        return 0
    execution_path = OUTPUT / "execution_manifest.json"
    if execution_path.exists():
        raise SystemExit(f"ablation already started; refusing a second run: {execution_path}")
    write_json(
        execution_path,
        {
            "schema_version": "gpt56luna_gate_ablation_execution.v1",
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
            print(json.dumps({"event": "gate_ablation_cell_complete", **result}, ensure_ascii=False), flush=True)
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
