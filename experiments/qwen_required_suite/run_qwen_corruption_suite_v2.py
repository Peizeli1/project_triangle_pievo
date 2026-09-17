#!/usr/bin/env python3
"""Qwen3-235B controlled evaluator-corruption suite, execution-safe v2.

Compared arms: matched PiEvo vs full RADP on MBO.
Conditions: clean, small transient spike, large transient spike, persistent bias.
The only execution change from the initial dry-run design is a uniform
max_invalid_attempts=12 safeguard for both methods. Invalid proposals never
count toward the primary budget. Runs that end before eight valid primaries are
retained and classified as early_terminated rather than retried by score.
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

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/run_matched_qwen235b_task.sh"
BASH = Path("/opt/homebrew/bin/bash")
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v11.json"
DECODE = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"
MODEL = "qwen3-235b-a22b-instruct-2507"
MODEL_TRACK = "matched_qwen235b"
TASK = "MBO"
METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)
BUDGET = 8
MAX_INVALID_ATTEMPTS = 12
CONDITIONS: dict[str, dict[str, str]] = {
    "clean": {"mode": "none", "calls": "", "offset": "0.0"},
    "transient_small": {"mode": "one_shot_spike", "calls": "3", "offset": "1.0"},
    "transient_large": {"mode": "one_shot_spike", "calls": "3", "offset": "2.5"},
    "persistent_bias": {"mode": "persistent_bias", "calls": "3,5", "offset": "1.5"},
}
OUTPUT = ROOT / "experiments/results/qwen_required_suite_v2/corruption"
PLAN = OUTPUT / "corruption_protocol_v2.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cells() -> list[str]:
    return [f"{condition}:{method}:{seed}" for condition in CONDITIONS for method in METHODS for seed in SEEDS]


def expected_plan() -> dict[str, Any]:
    return {
        "schema_version": "qwen_required_corruption.v2",
        "model": MODEL,
        "model_track": MODEL_TRACK,
        "task": TASK,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": BUDGET,
        "max_invalid_attempts": MAX_INVALID_ATTEMPTS,
        "invalid_attempts_count_toward_primary_budget": False,
        "conditions": CONDITIONS,
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "decode_path": str(DECODE.resolve()),
        "decode_sha256": sha256(DECODE),
        "runner_sha256": sha256(RUNNER),
        "early_termination_policy": "retain the run; for best-so-far trajectory summaries carry the last observed best forward to the nominal budget; never synthesize unevaluated primary outcomes",
        "selection_rule": "retain every predeclared condition-method-seed cell; no score-based retry, deletion, or seed replacement",
        "claim_boundary": "controlled evaluator-side corruption only; persistent bias is a proxy/evaluator-failure stress test and is not labeled as naturally occurring reward hacking",
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
    condition, method, seed_text = spec.split(":")
    seed = int(seed_text)
    cfg = CONDITIONS[condition]
    out = OUTPUT / f"qwen235b_{method}_mbo_seed{seed}_budget{BUDGET}_{condition}"
    if out.exists() and any(out.iterdir()):
        return {"cell": spec, "status": "refused_nonfresh_output", "returncode": 3, "output": str(out)}
    env = os.environ.copy()
    env.pop("ZJU_TASK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update({
        "PIEVO_TASK": TASK,
        "PIEVO_METHOD": method,
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
    if method == "radp":
        env.update({
            "PIEVO_RADP_SEARCH_STRATEGY": "afe_v9",
            "PIEVO_SOURCE_GATE_BEHAVIOR": "fixed",
        })
    process = subprocess.run(
        [str(BASH), str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.log").write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8")
    valid_primary = count_valid_primary(out)
    if process.returncode == 0 and valid_primary == BUDGET:
        status = "completed"
    elif process.returncode == 10 and 0 < valid_primary < BUDGET:
        status = "early_terminated"
    else:
        status = "failed"
    result = {
        "cell": spec,
        "status": status,
        "returncode": process.returncode,
        "valid_primary": valid_primary,
        "output": str(out.resolve()),
    }
    write_json(out / "corruption_cell_attestation.json", {
        "schema_version": "qwen_required_corruption_cell.v2",
        **result,
        "condition": condition,
        "condition_config": cfg,
        "method": method,
        "seed": seed,
        "max_invalid_attempts": MAX_INVALID_ATTEMPTS,
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
    selected = [f"{c}:{m}:{s}" for c in selected_conditions for m in METHODS for s in SEEDS]
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "cells": len(cells()), "selected": selected, "plan": str(PLAN)}, indent=2))
        return 0
    batch_name = "_".join(selected_conditions)
    status_path = OUTPUT / f"batch_status_{batch_name}.json"
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
    write_json(status_path, {
        "schema_version": "qwen_required_corruption_batch.v2",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": list(selected_conditions),
        "results": results,
        "completed": sum(r["status"] == "completed" for r in results),
        "early_terminated": sum(r["status"] == "early_terminated" for r in results),
        "failed": sum(r["status"] not in {"completed", "early_terminated"} for r in results),
    })
    return 0 if all(r["status"] in {"completed", "early_terminated"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
