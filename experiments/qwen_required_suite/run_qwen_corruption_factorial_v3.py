#!/usr/bin/env python3
"""Complete the Qwen corruption 2x2 AFE x gate factorial.

The existing v2 suite contains PiEvo (legacy search, gate off) and full RADP
(AFE search, gate on).  This runner collects only the two missing arms under
the same model, evaluator, conditions, seeds, and primary budget:

* triage_only: legacy search, gate on
* radp_no_gate: AFE search, gate off

Existing v2 artifacts are never modified or rerun by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.qwen_required_suite.run_qwen_factorial import arm_env

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/run_matched_qwen235b_task.sh"
BASH = Path("/opt/homebrew/bin/bash")
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v11.json"
DECODE = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"
MODEL = "qwen3-235b-a22b-instruct-2507"
MODEL_TRACK = "matched_qwen235b"
TASK = "MBO"
ARMS = ("triage_only", "radp_no_gate")
SEEDS = (0, 1, 2)
BUDGET = 8
MAX_INVALID_ATTEMPTS = 12
CONDITIONS: dict[str, dict[str, str]] = {
    "clean": {"mode": "none", "calls": "", "offset": "0.0"},
    "transient_small": {"mode": "one_shot_spike", "calls": "3", "offset": "1.0"},
    "transient_large": {"mode": "one_shot_spike", "calls": "3", "offset": "2.5"},
    "persistent_bias": {"mode": "persistent_bias", "calls": "3,5", "offset": "1.5"},
}
OUTPUT = ROOT / "experiments/results/qwen_required_suite_v3/corruption_factorial"
PLAN = OUTPUT / "corruption_factorial_protocol_v3.json"
PRIOR = ROOT / "experiments/results/qwen_required_suite_v2/corruption"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def selected_cells() -> list[str]:
    return [f"{condition}:{arm}:{seed}" for condition in CONDITIONS for arm in ARMS for seed in SEEDS]


def expected_plan() -> dict[str, Any]:
    prior_protocol = PRIOR / "corruption_protocol_v2.json"
    return {
        "schema_version": "qwen_required_corruption_factorial.v3",
        "objective": "complete the missing arms of the controlled 2x2 AFE-by-gate corruption factorial",
        "model": MODEL,
        "model_track": MODEL_TRACK,
        "task": TASK,
        "new_arms": list(ARMS),
        "full_factorial_arms": ["pievo", "triage_only", "radp_no_gate", "radp_full"],
        "seeds": list(SEEDS),
        "primary_budget_per_cell": BUDGET,
        "max_invalid_attempts": MAX_INVALID_ATTEMPTS,
        "conditions": CONDITIONS,
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "decode_path": str(DECODE.resolve()),
        "decode_sha256": sha256(DECODE),
        "runner_sha256": sha256(RUNNER),
        "prior_two_arm_protocol": str(prior_protocol.resolve()),
        "prior_two_arm_protocol_sha256": sha256(prior_protocol),
        "selection_rule": "retain every predeclared condition-arm-seed cell; no outcome-dependent retry, deletion, or seed replacement",
        "early_termination_policy": "retain and report; never synthesize unevaluated primary outcomes",
        "selected_cells": selected_cells(),
    }


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = expected_plan()
    if not PLAN.exists():
        write_json(PLAN, expected)
    observed = json.loads(PLAN.read_text(encoding="utf-8"))
    if observed != expected:
        raise SystemExit(f"frozen plan mismatch: {PLAN}")


def count_valid_primary(out: Path) -> int:
    path = out / "official_chembl_tool_audit.jsonl"
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        value = row.get("output")
        if (
            str(row.get("tool_name", "")).startswith("evaluate_")
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            count += 1
    return count


def run_cell(spec: str) -> dict[str, Any]:
    condition, arm, seed_text = spec.split(":")
    seed = int(seed_text)
    cfg = CONDITIONS[condition]
    out = OUTPUT / f"qwen235b_{arm}_mbo_seed{seed}_budget{BUDGET}_{condition}"
    if out.exists() and any(out.iterdir()):
        valid = count_valid_primary(out)
        status = "completed_existing" if valid == BUDGET else "refused_nonfresh_output"
        return {"cell": spec, "status": status, "returncode": 0 if valid == BUDGET else 3, "valid_primary": valid, "output": str(out)}

    env = os.environ.copy()
    env.pop("ZJU_TASK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update({
        "PIEVO_TASK": TASK,
        "PIEVO_SEED": str(seed),
        "PIEVO_BUDGET": str(BUDGET),
        "PIEVO_RUN_STAGE": "development",
        "PIEVO_RUN_TRACK": f"{MODEL_TRACK}_reconstruction",
        "PIEVO_MODEL_TRACK": MODEL_TRACK,
        "PIEVO_AGENT_MODEL": MODEL,
        "PIEVO_PROTOCOL_PATH": str(PROTOCOL.resolve()),
        "PIEVO_DECODE_PROTOCOL": str(DECODE.resolve()),
        "PIEVO_OUT": str(out.resolve()),
        "PIEVO_MAX_TOTAL_TOKENS": "150000",
        "PIEVO_AGENT_MAX_TOKENS": "4096",
        "PIEVO_MAX_RETRIES": "0",
        "PIEVO_MAX_INVALID_ATTEMPTS": str(MAX_INVALID_ATTEMPTS),
        "PIEVO_CORRUPTION_MODE": cfg["mode"],
        "PIEVO_CORRUPTION_CALLS": cfg["calls"],
        "PIEVO_CORRUPTION_OFFSET": cfg["offset"],
        "PIEVO_SOURCE_GATE_SCORE_GE": "0.0",
        "PIEVO_PAPER_ELIGIBLE": "0",
    })
    env.update(arm_env(TASK, arm))
    process = subprocess.run(
        [str(BASH), str(RUNNER)], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.log").write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8")
    valid = count_valid_primary(out)
    if process.returncode == 0 and valid == BUDGET:
        status = "completed"
    elif process.returncode == 10 and 0 < valid < BUDGET:
        status = "early_terminated"
    else:
        status = "failed"
    result = {"cell": spec, "status": status, "returncode": process.returncode, "valid_primary": valid, "output": str(out.resolve())}
    write_json(out / "corruption_factorial_cell_attestation.json", {
        "schema_version": "qwen_required_corruption_factorial_cell.v3",
        **result,
        "condition": condition,
        "condition_config": cfg,
        "arm": arm,
        "seed": seed,
        "plan_path": str(PLAN.resolve()),
        "plan_sha256": sha256(PLAN),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--conditions", nargs="*", choices=list(CONDITIONS))
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    prepare()
    selected_conditions = tuple(args.conditions) if args.conditions else tuple(CONDITIONS)
    selected = [f"{c}:{a}:{s}" for c in selected_conditions for a in ARMS for s in SEEDS]
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "new_cells": len(selected), "plan": str(PLAN)}, indent=2))
        return 0
    status_path = OUTPUT / f"batch_status_{'_'.join(selected_conditions)}.json"
    if status_path.exists():
        raise SystemExit(f"batch status already exists: {status_path}")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(run_cell, spec): spec for spec in selected}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: item["cell"])
    accepted = {"completed", "completed_existing", "early_terminated"}
    write_json(status_path, {
        "schema_version": "qwen_required_corruption_factorial_batch.v3",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": list(selected_conditions),
        "results": results,
        "completed": sum(r["status"] in {"completed", "completed_existing"} for r in results),
        "early_terminated": sum(r["status"] == "early_terminated" for r in results),
        "failed": sum(r["status"] not in accepted for r in results),
    })
    return 0 if all(r["status"] in accepted for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
