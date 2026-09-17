#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runtime_credentials import read_api_key

MODEL = "openai/qwen3-235b-a22b-instruct-2507"
API_BASE = "https://zjuapi.com/v1"
AGENT = "experiments.external_harness_pilot.stateful_triage.harbor_triage_agent:StatefulTriageTerminus"
PYTHON = "/Users/michael/.local/bin/python3.12"
TASK_ROOT = Path("experiments/external_harness_pilot/AARRI-pilot/tasks")
RESULT_ROOT = Path("experiments/results/qwen_required_suite_v2/aarri_stateful_v2")
EXEC_RULE = Path("experiments/external_harness_pilot/aarri_execution_rule.md")

FROZEN_TASKS = [
    "benchmark-split-ambiguity",
    "silent-eval-contamination",
    "unfair-baseline-sabotage",
    "log-buried-admission",
    "p-hacking-multitest",
]


def newest_result(job_dir: Path) -> Path | None:
    candidates = sorted(job_dir.glob("*/result.json"))
    return candidates[-1] if candidates else None


def run_task(task: str) -> dict:
    if task not in FROZEN_TASKS:
        raise ValueError(f"task {task!r} is outside the frozen subset")
    task_path = TASK_ROOT / task
    if not task_path.is_dir():
        raise FileNotFoundError(task_path)

    out = RESULT_ROOT / f"stateful_{task}"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    env["OPENAI_API_KEY"] = read_api_key()

    cmd = [
        "uv", "run", "--python", PYTHON, "--with", "harbor==0.21.0",
        "harbor", "run",
        "-p", str(task_path),
        "-a", AGENT,
        "-m", MODEL,
        "--ak", f"api_base={API_BASE}",
        "--ak", "max_turns=40",
        "--ak", "record_terminal_session=false",
        "--extra-instruction-path", str(EXEC_RULE),
        "--agent-setup-timeout-multiplier", "5",
        "--environment-build-timeout-multiplier", "5",
        "--verifier-timeout-multiplier", "5",
        "--n-concurrent", "1",
        "--max-retries", "0",
        "-o", str(out),
        "-y",
    ]
    completed = subprocess.run(cmd, env=env, text=True)
    result_path = newest_result(out)
    summary = {
        "task": task,
        "returncode": completed.returncode,
        "result_path": str(result_path) if result_path else None,
    }
    if result_path:
        try:
            result = json.loads(result_path.read_text())
            summary["job_result"] = result
        except Exception as exc:
            summary["result_parse_error"] = str(exc)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*", choices=FROZEN_TASKS)
    args = p.parse_args()
    tasks = args.tasks or FROZEN_TASKS
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL,
        "api_base": API_BASE,
        "agent": AGENT,
        "harbor": "0.21.0",
        "frozen_tasks": FROZEN_TASKS,
        "requested_tasks": tasks,
        "max_turns": 40,
        "execution_rule": str(EXEC_RULE),
        "tool_contract": "embedded in StatefulTriageTerminus",
        "results": [],
    }
    for task in tasks:
        manifest["results"].append(run_task(task))
        (RESULT_ROOT / "execution_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
