#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv-researchharness/bin/python"
RUNNER = ROOT / "experiments/external_harness_pilot/run_researchharness_triage_smoke.py"
MODEL = "qwen3-235b-a22b-instruct-2507"
CASES = ("noise", "mechanism")
ARMS = ("vanilla", "self_review", "gate")
OUT = ROOT / "experiments/results/researchharness_three_arm_20260901"


def extract_revise_allowed(stdout: str) -> bool | None:
    matches = re.findall(r'\\?"revise_allowed\\?"\s*:\s*(true|false|\\?"(?:yes|no)\\?")', stdout, flags=re.I)
    if matches:
        value = matches[-1].lower().strip('\\"')
        return value in {"true", "yes"}
    lowered = stdout.lower()
    if 'revise_allowed' in lowered:
        tail = lowered.rsplit('revise_allowed', 1)[-1][:100]
        if 'false' in tail or 'no' in tail:
            return False
        if 'true' in tail or 'yes' in tail:
            return True
    return None


def run_one(case: str, arm: str, repeat: int) -> dict[str, Any]:
    trace = OUT / "traces" / f"{case}_{arm}_repeat{repeat}"
    cmd = [
        str(PYTHON), str(RUNNER),
        "--model", MODEL,
        "--workspace", str(ROOT / f"experiments/external_harness_pilot/{case}_case"),
        "--trace-dir", str(trace),
    ]
    if arm == "gate":
        cmd.append("--with-gate")
    elif arm == "self_review":
        cmd.append("--self-review")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return {
        "case": case,
        "arm": arm,
        "repeat": repeat,
        "returncode": p.returncode,
        "revise_allowed": extract_revise_allowed(p.stdout),
        "stdout": p.stdout,
        "stderr": p.stderr,
        "trace_dir": str(trace),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [(c, a, r) for r in range(args.repeats) for c in CASES for a in ARMS]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(run_one, *spec): spec for spec in specs}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps({k: row[k] for k in ("case", "arm", "repeat", "returncode", "revise_allowed")}, ensure_ascii=False), flush=True)
    results.sort(key=lambda r: (r["repeat"], r["case"], r["arm"]))
    summary: dict[str, Any] = {}
    for case in CASES:
        summary[case] = {}
        expected = case == "mechanism"
        for arm in ARMS:
            rows = [r for r in results if r["case"] == case and r["arm"] == arm]
            parsed = [r["revise_allowed"] for r in rows if isinstance(r["revise_allowed"], bool)]
            summary[case][arm] = {
                "n": len(rows),
                "parsed": len(parsed),
                "revision_rate": sum(bool(x) for x in parsed) / len(parsed) if parsed else None,
                "decision_accuracy": sum(x == expected for x in parsed) / len(parsed) if parsed else None,
            }
    payload = {
        "schema_version": "qwen_researchharness_controlled.v2",
        "model": MODEL,
        "repeats": args.repeats,
        "cases": {
            "noise": "primary conflicts with repeat and independent verifier; correct action is reject revision",
            "mechanism": "primary agrees with repeat and independent verifier; correct action is allow revision",
        },
        "results": results,
        "summary": summary,
        "claim_boundary": "Controlled ResearchHarness evidence-admission test only; not an official ResearchClawBench leaderboard run.",
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if all(r["returncode"] == 0 and isinstance(r["revise_allowed"], bool) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
