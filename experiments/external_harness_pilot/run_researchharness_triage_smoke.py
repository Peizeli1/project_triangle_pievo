#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from researchharness import Read, run_agent, tool

from experiments.runtime_credentials import read_api_key
from experiments.source_aware_gate import SourceAwareGate


@tool(name="triage_evidence", description="Apply the source-aware evidence-admission gate to one anomalous observation before scientific belief revision.")
def triage_evidence(
    primary: float,
    repeat: float,
    independent: float,
    surprisal: float,
    threshold: float = 0.85,
) -> dict:
    gate = SourceAwareGate(tolerance=0.25)
    return asdict(
        gate.decide(
            primary=primary,
            repeat=repeat,
            independent=independent,
            surprisal=surprisal,
            threshold=threshold,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--with-gate", action="store_true")
    parser.add_argument("--self-review", action="store_true")
    parser.add_argument("--workspace", default="experiments/external_harness_pilot/smoke_workspace")
    parser.add_argument("--trace-dir", default="experiments/external_harness_pilot/traces")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if args.with_gate and args.self_review:
        parser.error("--with-gate and --self-review are mutually exclusive")
    arm = "gate" if args.with_gate else ("self_review" if args.self_review else "vanilla")
    trace_dir = Path(args.trace_dir).resolve() / (args.model.replace("/", "_") + f"_{arm}")
    trace_dir.mkdir(parents=True, exist_ok=True)

    tools = [Read, triage_evidence] if args.with_gate else [Read]
    role = (
        "You are an autonomous scientific-research agent. Treat anomalous observations as evidence, not automatic permission to revise scientific beliefs. "
        "Use only the evidence available in the workspace and exposed tools. Return a concise JSON object with keys decision, revise_allowed, rationale."
    )
    if args.with_gate:
        role += " For anomalous numerical evidence, you must call triage_evidence before deciding whether revision is allowed."
    elif args.self_review:
        role += (
            " Before returning the final JSON, independently critique your initial decision, "
            "look for evidence that would reverse it, and correct the decision if needed."
        )

    prompt = (
        "Read observation.txt. The current prediction is 5.7, so compute surprisal as abs(primary-current_prediction). "
        "Decide whether the anomalous primary observation should be admitted to scientific belief revision."
    )

    answer = run_agent(
        prompt,
        workspace_root=str(workspace),
        model_name=args.model,
        api_key=read_api_key(),
        api_base="https://zjuapi.com/v1",
        role_prompt=role,
        tools=tools,
        require_env=False,
        max_rounds=8,
        max_runtime_seconds=180,
        max_input_tokens=32768,
        max_output_tokens=2048,
        omit_generate_params=["presence_penalty"],
        trace_dir=str(trace_dir),
    )
    print(json.dumps({"model": args.model, "arm": arm, "answer": answer}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
