#!/usr/bin/env python3
"""Recover only failed AARRI trials after a full frozen job reaches terminal state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runtime_credentials import read_api_key
from experiments.external_harness_pilot.stateful_triage.run_aarri_formal_matrix import (
    AGENTS,
    API_BASE,
    EXEC_RULE,
    HARBOR_VERSION,
    MAX_TURNS,
    MODEL,
    PYTHON,
    RESULT_ROOT,
    SEEDS,
    TASK_ROOT,
    TEMPERATURE,
)


def records(job_root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in job_root.glob("**/result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task = data.get("task_name")
        if task and data.get("trial_name"):
            out.setdefault(task, []).append(data)
    return out


def has_valid_result(rows: list[dict]) -> bool:
    return any(
        row.get("exception_info") is None
        and (row.get("verifier_result") or {}).get("rewards", {}).get("reward") is not None
        for row in rows
    )


def recover(arm: str, seed: int, task: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["OPENAI_API_KEY"] = read_api_key()
    llm_call_kwargs = json.dumps({"seed": seed}, separators=(",", ":"))
    output = RESULT_ROOT / f"{arm}_seed{seed}" / "recovery" / task
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "--python", PYTHON, "--with", f"harbor=={HARBOR_VERSION}",
        "harbor", "run", "-p", str(TASK_ROOT / task), "-a", AGENTS[arm], "-m", MODEL,
        "--job-name", f"recover_{arm}_seed{seed}_{task}",
        "--ak", f"api_base={API_BASE}", "--ak", f"max_turns={MAX_TURNS}",
        "--ak", f"temperature={TEMPERATURE}", "--ak", f"llm_call_kwargs={llm_call_kwargs}",
        "--ak", "record_terminal_session=false", "--ak", "store_all_messages=false",
        "--extra-instruction-path", str(EXEC_RULE),
        "--agent-setup-timeout-multiplier", "5", "--environment-build-timeout-multiplier", "5",
        "--verifier-timeout-multiplier", "5", "--n-concurrent", "1", "--max-retries", "0",
        "-o", str(output), "-y",
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True).returncode


def main() -> int:
    attempted = 0
    unresolved = []
    for seed in SEEDS:
        for arm in AGENTS:
            job_root = RESULT_ROOT / f"{arm}_seed{seed}"
            before = records(job_root)
            if len(before) != 82:
                continue
            failed = [task for task, rows in before.items() if not has_valid_result(rows)]
            for task in failed:
                attempted += 1
                recover(arm, seed, task)
            after = records(job_root)
            unresolved.extend(
                {"arm": arm, "seed": seed, "task": task}
                for task, rows in after.items()
                if not has_valid_result(rows)
            )
    status = {"recovery_attempts": attempted, "unresolved": unresolved}
    (RESULT_ROOT / "recovery_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status))
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
