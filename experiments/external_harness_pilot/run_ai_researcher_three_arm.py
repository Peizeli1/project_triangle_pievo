#!/usr/bin/env python3
"""Run HKUDS AI-Researcher's MetaChain runtime on the matched three-arm study."""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AI_RESEARCHER = ROOT / "official_sources/AI-Researcher"
sys.path.insert(0, str(AI_RESEARCHER))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEFAULT_LOG", "0")
os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("API_BASE_URL", "https://zjuapi.com/v1")

from experiments.runtime_credentials import read_api_key
from experiments.source_aware_gate import SourceAwareGate
from research_agent.inno import MetaChain
from research_agent.inno.types import Agent
import research_agent.inno.core as metachain_core

MODEL = "openai/qwen3-235b-a22b-instruct-2507"
CASES = ("noise", "mechanism")
ARMS = ("base", "self_review", "triage")
OUT = ROOT / "experiments/results/ai_researcher_three_arm_20260901"


def triage_evidence(
    primary: float,
    repeat: float,
    independent: float,
    prediction: float,
    threshold: float,
) -> str:
    """Verify an anomalous observation and return an auditable admission decision."""
    decision = SourceAwareGate(tolerance=0.25).decide(
        primary=primary,
        repeat=repeat,
        independent=independent,
        surprisal=abs(primary - prediction),
        threshold=threshold,
    )
    return json.dumps(asdict(decision), sort_keys=True)


def extract_decision(text: str) -> bool | None:
    matches = re.findall(r'"revise_allowed"\s*:\s*(true|false)', text, flags=re.I)
    return matches[-1].lower() == "true" if matches else None


def run_one(case: str, arm: str, repeat: int) -> dict[str, Any]:
    evidence = (ROOT / f"experiments/external_harness_pilot/{case}_case/observation.txt").read_text(encoding="utf-8")
    instructions = (
        "You are an autonomous scientific-research agent deciding whether anomalous numerical evidence may revise a persistent belief. "
        "Return only a JSON object with keys decision, revise_allowed, and rationale; revise_allowed must be a JSON boolean."
    )
    functions = []
    if arm == "self_review":
        instructions += " Silently critique the initial decision against all supplied evidence and correct it before returning JSON."
    elif arm == "triage":
        instructions += (
            " You must call triage_evidence with the supplied numerical values before deciding. "
            "Copy its update_allowed boolean into the final revise_allowed field."
        )
        functions = [triage_evidence]
    agent = Agent(name=f"AI-Researcher-{arm}", model=MODEL, instructions=instructions, functions=functions)

    calls: list[dict[str, Any]] = []
    original_completion = metachain_core.completion

    def recorded_completion(*args: Any, **kwargs: Any) -> Any:
        response = original_completion(*args, **kwargs)
        usage = getattr(response, "usage", None)
        calls.append(usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {}))
        return response

    metachain_core.completion = recorded_completion
    started = time.perf_counter()
    error = None
    text = ""
    try:
        response = MetaChain().run(
            agent,
            [{"role": "user", "content": f"Evidence:\n{evidence}"}],
            debug=False,
            max_turns=3 if arm == "triage" else 1,
        )
        text = str(response.messages[-1].get("content") or "")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        metachain_core.completion = original_completion
    runtime = time.perf_counter() - started
    total_tokens = sum(int(call.get("total_tokens") or 0) for call in calls)
    return {
        "case": case,
        "arm": arm,
        "repeat": repeat,
        "revise_allowed": extract_decision(text),
        "result_text": text,
        "runtime_seconds": runtime,
        "total_tokens": total_tokens,
        "llm_calls": calls,
        "tool_required": arm == "triage",
        "tool_called": len(calls) >= 2 if arm == "triage" else None,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    os.environ["OPENAI_API_KEY"] = read_api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.repeats):
        for case in CASES:
            for arm in ARMS:
                row = run_one(case, arm, repeat)
                rows.append(row)
                (OUT / "results.json").write_text(
                    json.dumps({"implementation": "HKUDS/AI-Researcher MetaChain", "revision": "origin/main", "model": MODEL, "results": rows}, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps({k: row[k] for k in ("case", "arm", "repeat", "revise_allowed", "runtime_seconds", "total_tokens", "tool_called", "error")}), flush=True)
    failures = [row for row in rows if row["error"] or not isinstance(row["revise_allowed"], bool) or (row["arm"] == "triage" and not row["tool_called"])]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
