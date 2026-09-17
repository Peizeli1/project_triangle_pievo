#!/usr/bin/env python3
"""Run a small, single-model NewtonBench pilot with Luna no-thinking.

The credential is read from the local Keychain and passed only to the child
process environment.  It is never written to the benchmark directory or to
the pilot manifest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from runtime_credentials import read_api_key


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "official_sources/NewtonBench"
PYTHON = ROOT / ".venv/bin/python"
MANIFEST = ROOT / "experiments/results/newtonbench_gpt56luna_nothinking_pilot_20260814.json"


def main() -> int:
    command = [
        str(PYTHON),
        "run_all_evaluations.py",
        "--model_name",
        "gpt-5.6-luna",
        "--module",
        "m0_gravity",
        "--noise",
        "0",
        "--trials_per_law",
        "3",
        "--equation_difficulty",
        "hard",
        "--model_system",
        "complex_system",
        "--agent_backend",
        "code_assisted_agent",
        "--no_prompt",
    ]
    key = read_api_key(None)
    env = os.environ.copy()
    env["ZJU_TASK_API_KEY"] = key
    env["ZJU_API_BASE_URL"] = "https://zjuapi.com/v1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(BENCHMARK), env.get("PYTHONPATH", "")]
    )
    env["PATH"] = os.pathsep.join([str(PYTHON.parent), env.get("PATH", "")])
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=BENCHMARK, env=env, text=True)
    payload = {
        "schema_version": "newtonbench_single_model_pilot.v1",
        "benchmark": "NewtonBench",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "enable_thinking": False,
        "api_route": "ZJU OpenAI-compatible gateway",
        "module": "m0_gravity",
        "noise": 0.0,
        "trials_per_law": 3,
        "equation_difficulty": "hard",
        "model_system": "complex_system",
        "agent_backend": "code_assisted_agent",
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "results_root": str((BENCHMARK / "evaluation_results/gpt-5.6-luna").resolve()),
        "claim_boundary": "Single-model NewtonBench pilot; not a full benchmark leaderboard or SOTA claim.",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
