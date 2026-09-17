#!/usr/bin/env python3
"""Run the predeclared Luna formal comparison extension for seeds 1 and 2."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "experiments/run_gpt56luna_pievo_radp_cell.sh"
PROTOCOL = ROOT / "experiments/configs/gpt56luna_medium_pievo_radp_protocol_v1.json"
OUTPUT = ROOT / "experiments/results/gpt56luna_formal_seed12_20260812"
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (1, 2)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cell_specs() -> list[tuple[str, str, int]]:
    return [(method, task, seed) for task in TASKS for method in METHODS for seed in SEEDS]


def run_cell(spec: tuple[str, str, int]) -> dict[str, Any]:
    method, task, seed = spec
    cell = f"{method}:{task}:{seed}"
    out = OUTPUT / f"gpt56luna_{method}_{task.lower()}_seed{seed}_formal24"
    if out.exists() and any(out.iterdir()):
        return {"cell": cell, "status": "refused_nonfresh_output", "returncode": 3, "output": str(out)}
    env = os.environ.copy()
    env.pop("ZJU_TASK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update(
        {
            "PIEVO_TASK": task,
            "PIEVO_METHOD": method,
            "PIEVO_SEED": str(seed),
            "PIEVO_BUDGET": "24",
            "PIEVO_OUT": str(out.resolve()),
            "PIEVO_RUN_STAGE": "formal",
            "PIEVO_RUN_TRACK": "gpt56luna",
            "PIEVO_MODEL_TRACK": "gpt56luna",
            "PIEVO_AGENT_MODEL": "gpt-5.6-luna",
            "PIEVO_REASONING_EFFORT": "medium",
            "PIEVO_PROTOCOL_PATH": str(PROTOCOL.resolve()),
            "PIEVO_DECODE_PROTOCOL": str(PROTOCOL.resolve()),
        }
    )
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{method}_{task.lower()}_seed{seed}.log"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            ["bash", str(WRAPPER)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True
        )
    manifest = out / "run_manifest.json"
    summary = {}
    if manifest.is_file():
        try:
            summary = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    return {
        "cell": cell,
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "output": str(out.resolve()),
        "manifest_status": summary.get("status"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / "formal_seed12_protocol.json"
    plan = {
        "schema_version": "gpt56luna_formal_seed12_extension.v1",
        "paper_eligible": True,
        "purpose": "complete_predeclared_three_seed_formal_comparison",
        "model": "gpt-5.6-luna",
        "model_track": "gpt56luna",
        "reasoning_effort": "medium",
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": 24,
        "selected_cells": [f"{m}:{t}:{s}" for t, m, s in cell_specs()],
        "protocol_path": str(PROTOCOL.resolve()),
        "claim_boundary": "Formal three-seed matched released-evaluator extension; not universal external SOTA.",
    }
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise SystemExit(f"formal seed-1/2 plan differs: {plan_path}")
    else:
        write_json(plan_path, plan)
    execution_path = OUTPUT / "execution_manifest.json"
    if execution_path.exists():
        raise SystemExit(f"formal seed-1/2 extension already started: {execution_path}")
    write_json(
        execution_path,
        {
            "schema_version": "gpt56luna_formal_seed12_execution.v1",
            "plan_path": str(plan_path.resolve()),
            "selected_cells": plan["selected_cells"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        },
    )
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_cell, spec): spec for spec in cell_specs()}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "formal_seed12_cell_complete", **result}, ensure_ascii=False), flush=True)
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
