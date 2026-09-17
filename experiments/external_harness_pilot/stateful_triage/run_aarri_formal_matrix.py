#!/usr/bin/env python3
"""Run the frozen 82-task AARRI Base/Self-Review/Stateful-Triage matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runtime_credentials import read_api_key

MODEL = "openai/qwen3-235b-a22b-instruct-2507"
API_BASE = "https://zjuapi.com/v1"
PYTHON = "/Users/michael/.local/bin/python3.12"
HARBOR_VERSION = "0.21.0"
TASK_ROOT = PROJECT_ROOT / "experiments/external_harness_pilot/AARRI-bench/tasks"
RESULT_ROOT = PROJECT_ROOT / "experiments/results/aarri_formal_matrix_20260903"
EXEC_RULE = PROJECT_ROOT / "experiments/external_harness_pilot/aarri_execution_rule.md"
EXPECTED_COMMIT = "6b0d231eb0d2e9b5f10b1cc225bf61fba3cf5ed2"
SEEDS = (0, 1, 2)
TEMPERATURE = 0.6
MAX_TURNS = 40

AGENTS = {
    "base": "terminus-2",
    "self_review": (
        "experiments.external_harness_pilot.stateful_triage.harbor_triage_agent:"
        "SelfReviewTerminus"
    ),
    "stateful_triage": (
        "experiments.external_harness_pilot.stateful_triage.harbor_triage_agent:"
        "StatefulTriageTerminus"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_manifest() -> list[dict]:
    rows = []
    for path in sorted(TASK_ROOT.glob("*/task.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        environment = data.get("environment", {})
        rows.append(
            {
                "task": path.parent.name,
                "task_id": metadata.get("task_id"),
                "category": metadata.get("category"),
                "difficulty": metadata.get("difficulty"),
                "gpus": environment.get("gpus", 0),
                "cpus": environment.get("cpus"),
                "memory_mb": environment.get("memory_mb"),
            }
        )
    if len(rows) != 82:
        raise RuntimeError(f"expected 82 frozen AARRI tasks, found {len(rows)}")
    if any(row["gpus"] not in (0, None) for row in rows):
        raise RuntimeError("the frozen task set contains a task requesting a GPU")
    return rows


def upstream_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(TASK_ROOT.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def trial_results(job_root: Path) -> list[Path]:
    paths = []
    for path in job_root.glob("**/result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("trial_name") and data.get("task_name"):
            paths.append(path)
    return paths


def terminal_tasks(job_root: Path) -> set[str]:
    tasks = set()
    for path in trial_results(job_root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("task_name"):
            tasks.add(data["task_name"])
    return tasks


def run_job(arm: str, seed: int, concurrency: int) -> dict:
    job_key = f"{arm}_seed{seed}"
    job_root = RESULT_ROOT / job_key
    marker = job_root / "formal_job_status.json"
    existing_tasks = terminal_tasks(job_root)
    if len(existing_tasks) == 82:
        print(json.dumps({"job": job_key, "status": "already_complete", "trials": 82}), flush=True)
        if marker.is_file():
            return json.loads(marker.read_text(encoding="utf-8"))
        return {
            "job": job_key,
            "arm": arm,
            "seed": seed,
            "returncode": None,
            "terminal_trials": 82,
            "expected_trials": 82,
            "complete": True,
        }

    job_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["OPENAI_API_KEY"] = read_api_key()
    llm_call_kwargs = json.dumps({"seed": seed}, separators=(",", ":"))
    cmd = [
        "uv", "run", "--python", PYTHON, "--with", f"harbor=={HARBOR_VERSION}",
        "harbor", "run",
        "-p", str(TASK_ROOT),
        "-a", AGENTS[arm],
        "-m", MODEL,
        "--job-name", job_key,
        "--ak", f"api_base={API_BASE}",
        "--ak", f"max_turns={MAX_TURNS}",
        "--ak", f"temperature={TEMPERATURE}",
        "--ak", f"llm_call_kwargs={llm_call_kwargs}",
        "--ak", "record_terminal_session=false",
        "--ak", "store_all_messages=false",
        "--extra-instruction-path", str(EXEC_RULE),
        "--agent-setup-timeout-multiplier", "5",
        "--environment-build-timeout-multiplier", "5",
        "--verifier-timeout-multiplier", "5",
        "--n-concurrent", str(concurrency),
        "--max-retries", "0",
        "-o", str(job_root),
        "-y",
    ]
    started = utc_now()
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True)
    found_tasks = terminal_tasks(job_root)
    status = {
        "job": job_key,
        "arm": arm,
        "seed": seed,
        "started_at": started,
        "finished_at": utc_now(),
        "returncode": completed.returncode,
        "terminal_trials": len(found_tasks),
        "expected_trials": 82,
        "complete": len(found_tasks) == 82,
    }
    marker.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=tuple(AGENTS), default=list(AGENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 3:
        parser.error("concurrency must be between 1 and 3 on this host")
    if any(seed not in SEEDS for seed in args.seeds):
        parser.error(f"seeds must be selected from {SEEDS}")

    commit = upstream_commit()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"AARRI revision changed: expected {EXPECTED_COMMIT}, found {commit}")
    tasks = task_manifest()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "benchmark": "AARRI-Bench",
        "benchmark_commit": commit,
        "task_count": len(tasks),
        "tasks": tasks,
        "model": MODEL,
        "api_base": API_BASE,
        "harbor": HARBOR_VERSION,
        "agents": AGENTS,
        "temperature": TEMPERATURE,
        "max_turns": MAX_TURNS,
        "seeds": list(SEEDS),
        "attempts_per_seed": 1,
        "local_gpu_required": False,
        "execution_rule": str(EXEC_RULE.relative_to(PROJECT_ROOT)),
        "created_at": utc_now(),
    }
    (RESULT_ROOT / "frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    statuses = []
    for seed in args.seeds:
        for arm in args.arms:
            statuses.append(run_job(arm, seed, args.concurrency))
    (RESULT_ROOT / "matrix_status.json").write_text(
        json.dumps({"jobs": statuses, "updated_at": utc_now()}, indent=2),
        encoding="utf-8",
    )
    return 0 if all(status["complete"] for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
