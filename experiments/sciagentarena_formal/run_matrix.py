#!/usr/bin/env python3
"""Run resumable SciAgentArena Table 6/7 cells and official scoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DD_ROOT = ROOT / "official_sources/SciAgentArena/evaluations/dd"
GENERATOR = ROOT / "experiments/sciagentarena_formal/harness_generate.py"
RESULT_ROOT = ROOT / "experiments/results/sciagentarena_drug_discovery_20260903"
SCORER_PYTHON = DD_ROOT / ".venv/bin/python"
INTERPRETERS = {
    "claude_code": Path(sys.executable),
    "codex_cli": Path(sys.executable),
    "direct_openai": SCORER_PYTHON,
    "ai_researcher": ROOT / ".venv-ai-researcher-controlled/bin/python",
    "researchharness": ROOT / ".venv-researchharness/bin/python",
}
SEEDS = (0, 1, 2)

TABLE6 = (
    ("claude_code", "base"),
    ("codex_cli", "base"),
    ("ai_researcher", "base"),
    ("researchharness", "base"),
    ("direct_openai", "triage_pievo"),
)
TABLE7 = tuple(
    (harness, variant)
    for harness in ("researchharness", "ai_researcher", "claude_code")
    for variant in ("base", "self_review", "triage")
)
TABLE8 = tuple(
    ("direct_openai", variant)
    for variant in (
        "triage_pievo", "triage_pievo_k1", "triage_pievo_k3",
        "triage_pievo_no_holdout", "triage_pievo_no_channel",
        "triage_pievo_no_backtrack", "triage_pievo_same_source",
    )
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tasks() -> list[dict[str, Any]]:
    rows = json.loads((DD_ROOT / "tasks_index.json").read_text(encoding="utf-8"))
    if len(rows) != 79:
        raise RuntimeError(f"expected 79 tasks, found {len(rows)}")
    return rows


def cell_dir(harness: str, variant: str, seed: int, task_id: str) -> Path:
    return RESULT_ROOT / "cells" / harness / variant / f"seed{seed}" / task_id


def terminal(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    # Completed cells are immutable. Failed attempts remain auditable but may
    # be retried after an environment or transient API repair.
    return data.get("status") == "complete"


def run_one(spec: tuple[str, str, int, dict[str, Any]]) -> dict[str, Any]:
    harness, variant, seed, task = spec
    out_dir = cell_dir(harness, variant, seed, task["task_id"])
    final_path = out_dir / "result.json"
    if terminal(final_path):
        return json.loads(final_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    generation_path = out_dir / "generation.json"
    code_path = out_dir / "solution.py"
    score_path = out_dir / "official_score.json"
    gen_cmd = [
        str(INTERPRETERS[harness]), str(GENERATOR),
        "--harness", harness, "--variant", variant,
        "--task", str(DD_ROOT / task["task_path"]),
        "--seed", str(seed), "--out", str(generation_path),
    ]
    started = now()
    try:
        generated = subprocess.run(gen_cmd, cwd=ROOT, text=True, capture_output=True, timeout=1500)
    except subprocess.TimeoutExpired as exc:
        generated = None
        generation_error = f"generation timeout: {exc}"
    else:
        generation_error = None if generated.returncode == 0 else generated.stderr[-3000:]
    generation = {}
    if generation_path.is_file():
        try:
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
        except Exception:
            generation = {}
    code = str(generation.get("code") or "")
    if not code:
        result = {
            "status": "generation_failed", "harness": harness, "variant": variant,
            "seed": seed, "task": task, "started_at": started, "finished_at": now(),
            "generation_error": generation_error, "generation": generation,
        }
        final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    code_path.write_text(code, encoding="utf-8")
    score_cmd = [
        str(SCORER_PYTHON), str(DD_ROOT / "evaluate.py"), "run",
        task["task_id"], str(code_path), "--out", str(score_path),
    ]
    try:
        scored = subprocess.run(score_cmd, cwd=DD_ROOT, text=True, capture_output=True, timeout=2100)
    except subprocess.TimeoutExpired as exc:
        scored = None
        scoring_error = f"scoring timeout: {exc}"
    else:
        scoring_error = None if scored.returncode == 0 else scored.stderr[-3000:]
    score = {}
    if score_path.is_file():
        try:
            score = json.loads(score_path.read_text(encoding="utf-8"))
        except Exception:
            score = {}
    status = "complete" if score.get("score") else "scoring_failed"
    result = {
        "status": status, "harness": harness, "variant": variant,
        "seed": seed, "task": task, "started_at": started, "finished_at": now(),
        "generation": generation, "official_score": score,
        "generation_error": generation_error, "scoring_error": scoring_error,
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", choices=("6", "7", "8"), default=["6", "7"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    selected_tasks = tasks()
    if args.task_ids:
        wanted = set(args.task_ids)
        selected_tasks = [task for task in selected_tasks if task["task_id"] in wanted]
        missing = wanted - {task["task_id"] for task in selected_tasks}
        if missing:
            parser.error("unknown task ids: " + ", ".join(sorted(missing)))
    methods = set()
    if "6" in args.tables:
        methods.update(TABLE6)
    if "7" in args.tables:
        methods.update(TABLE7)
    if "8" in args.tables:
        methods.update(TABLE8)
    specs = [
        (harness, variant, seed, task)
        for harness, variant in sorted(methods)
        for seed in args.seeds
        for task in selected_tasks
    ]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": now(), "protocol": "experiments/sciagentarena_formal/protocol.json",
        "tables": args.tables, "seeds": args.seeds, "task_count": len(selected_tasks),
        "methods": sorted([list(method) for method in methods]), "expected_cells": len(specs),
    }
    (RESULT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = {"complete": 0, "generation_failed": 0, "scoring_failed": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(run_one, spec): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            print(json.dumps({
                "done": sum(counts.values()), "total": len(specs), "status": result["status"],
                "harness": result["harness"], "variant": result["variant"],
                "seed": result["seed"], "task": result["task"]["task_id"],
            }), flush=True)
    (RESULT_ROOT / "matrix_status.json").write_text(
        json.dumps({"updated_at": now(), "counts": counts, "expected": len(specs)}, indent=2),
        encoding="utf-8",
    )
    return 0 if counts.get("complete") == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
