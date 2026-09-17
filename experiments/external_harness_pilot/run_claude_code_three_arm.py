#!/usr/bin/env python3
"""Run Claude Code Base/Self-Review/Triage on matched controlled cases."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.runtime_credentials import read_api_key

MODEL = "qwen3-235b-a22b-instruct-2507"
CASES = ("noise", "mechanism")
ARMS = ("base", "self_review", "triage")
TOOL = ROOT / "experiments/external_harness_pilot/controlled_triage_tool.py"
OUT = ROOT / "experiments/results/claude_code_three_arm_20260901"


def extract_decision(text: str) -> bool | None:
    matches = re.findall(r'"revise_allowed"\s*:\s*(true|false)', text, flags=re.I)
    return matches[-1].lower() == "true" if matches else None


def run_one(case: str, arm: str, repeat: int) -> dict[str, Any]:
    observation = (ROOT / f"experiments/external_harness_pilot/{case}_case/observation.txt").read_text(encoding="utf-8")
    prompt = f"""You are deciding whether anomalous numerical evidence may revise a persistent scientific belief.

Evidence:
{observation}

Return only a JSON object with keys decision, revise_allowed, and rationale. revise_allowed must be a JSON boolean."""
    allowed_tools = ""
    if arm == "self_review":
        prompt += "\nBefore the final JSON, silently critique the initial decision and correct it if contrary evidence exists."
    elif arm == "triage":
        prompt += (
            f"\nYou must run this gate before deciding, substituting the numerical values from the evidence: "
            f"python3 {TOOL} --primary VALUE --repeat VALUE --independent VALUE --prediction VALUE --threshold VALUE. "
            "Copy the returned update_allowed boolean into the final revise_allowed field."
        )
        allowed_tools = "Bash"

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = read_api_key()
    env["ANTHROPIC_BASE_URL"] = "https://zjuapi.com"
    cmd = [
        "claude", "--bare", "--print", "--model", MODEL,
        "--tools", allowed_tools,
        "--output-format", "json",
        "--no-session-persistence",
        "--max-budget-usd", "0.08",
    ]
    if arm == "triage":
        cmd.append("--dangerously-skip-permissions")
    cmd.append(prompt)
    completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pass
    result_text = str(parsed.get("result", "")) if parsed else completed.stdout
    record = {
        "case": case,
        "arm": arm,
        "repeat": repeat,
        "returncode": completed.returncode,
        "revise_allowed": extract_decision(result_text),
        "result_text": result_text,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_ms": parsed.get("duration_ms") if parsed else None,
        "duration_api_ms": parsed.get("duration_api_ms") if parsed else None,
        "total_cost_usd": parsed.get("total_cost_usd") if parsed else None,
        "usage": parsed.get("usage") if parsed else None,
        "model_usage": parsed.get("modelUsage") if parsed else None,
        "is_error": parsed.get("is_error") if parsed else None,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.repeats):
        for case in CASES:
            for arm in ARMS:
                row = run_one(case, arm, repeat)
                rows.append(row)
                (OUT / "results.json").write_text(json.dumps({"model": MODEL, "results": rows}, indent=2), encoding="utf-8")
                print(json.dumps({k: row[k] for k in ("case", "arm", "repeat", "returncode", "revise_allowed", "total_cost_usd")}), flush=True)
    failures = [row for row in rows if row["returncode"] != 0 or not isinstance(row["revise_allowed"], bool)]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
