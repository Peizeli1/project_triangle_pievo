#!/usr/bin/env python3
"""Run the predeclared GPT-5.6-Luna no-thinking formal matrix.

The matrix is deliberately stored in a new track directory.  It never reuses
or overwrites the earlier medium-reasoning Luna artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "experiments/run_gpt56luna_nothinking_pievo_radp_cell.sh"
PROTOCOL = ROOT / "experiments/configs/gpt56luna_nothinking_pievo_radp_protocol_v1.json"
OUTPUT = ROOT / "experiments/results/gpt56luna_nothinking_formal_20260813"
MODEL = "gpt-5.6-luna"
MODEL_TRACK = "gpt56luna_nothinking"
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)
BUDGET = 24


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cell_specs() -> list[tuple[str, str, int]]:
    return [(method, task, seed) for task in TASKS for method in METHODS for seed in SEEDS]


def run_cell(spec: tuple[str, str, int]) -> dict[str, Any]:
    method, task, seed = spec
    cell = f"{method}:{task}:{seed}"
    out = OUTPUT / f"{MODEL_TRACK}_{method}_{task.lower()}_seed{seed}_formal24"
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
            "PIEVO_BUDGET": str(BUDGET),
            "PIEVO_OUT": str(out.resolve()),
            "PIEVO_RUN_STAGE": "formal",
            "PIEVO_RUN_TRACK": MODEL_TRACK,
            "PIEVO_MODEL_TRACK": MODEL_TRACK,
            "PIEVO_AGENT_MODEL": MODEL,
            "PIEVO_AGENT_BASE_URL": "https://zjuapi.com/v1",
            "PIEVO_REASONING_EFFORT": "none",
            "PIEVO_PROTOCOL_PATH": str(PROTOCOL.resolve()),
            "PIEVO_DECODE_PROTOCOL": str(PROTOCOL.resolve()),
            "PIEVO_MBO_APPLICABILITY_DOMAIN": "none",
        }
    )
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{method}_{task.lower()}_seed{seed}.log"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            ["bash", str(WRAPPER)],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    manifest = out / "run_manifest.json"
    summary: dict[str, Any] = {}
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
    if not WRAPPER.is_file() or not PROTOCOL.is_file():
        raise SystemExit("no-thinking formal wrapper or protocol is missing")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"formal output directory must be fresh: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / "formal_matrix_protocol.json"
    plan = {
        "schema_version": "gpt56luna_nothinking_formal_matrix.v1",
        "paper_eligible": True,
        "purpose": "predeclared_three_seed_formal_matched_evaluator_comparison",
        "model": MODEL,
        "model_track": MODEL_TRACK,
        "reasoning_effort": "none",
        "enable_thinking": False,
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": BUDGET,
        "selected_cells": [f"{m}:{t}:{s}" for t, m, s in cell_specs()],
        "protocol_path": str(PROTOCOL.resolve()),
        "claim_boundary": "Formal three-seed matched released-evaluator comparison for the GPT-5.6-Luna no-thinking track; not universal external SOTA.",
    }
    write_json(plan_path, plan)
    execution_path = OUTPUT / "execution_manifest.json"
    write_json(
        execution_path,
        {
            "schema_version": "gpt56luna_nothinking_formal_execution.v1",
            "plan_path": str(plan_path.resolve()),
            "selected_cells": plan["selected_cells"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        },
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_cell, spec): spec for spec in cell_specs()}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "formal_nothinking_cell_complete", **result}, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: item["cell"])
    write_json(OUTPUT / "cell_status.json", results)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    failed = sum(item["status"] != "completed" for item in results)
    execution.update(
        {
            "status": "completed" if failed == 0 else "completed_with_failures",
            "completed_cells": len(results) - failed,
            "failed_cells": failed,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(execution_path, execution)
    return 0 if failed == 0 and len(results) == len(cell_specs()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
