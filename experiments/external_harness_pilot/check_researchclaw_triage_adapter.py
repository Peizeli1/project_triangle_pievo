#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict

from researchharness import tool
from experiments.source_aware_gate import SourceAwareGate


@tool(
    name="TriageEvidence",
    description=(
        "Verify an anomalous numerical observation before allowing it to change a scientific belief. "
        "Use only when primary, repeat, and independent held-out measurements are available."
    ),
)
def triage_evidence(
    primary: float,
    repeat: float,
    independent: float,
    prediction: float,
    threshold: float = 0.85,
) -> dict:
    gate = SourceAwareGate(tolerance=0.25)
    return asdict(
        gate.decide(
            primary=primary,
            repeat=repeat,
            independent=independent,
            surprisal=abs(primary - prediction),
            threshold=threshold,
        )
    )


def main() -> None:
    import benchmarks.ResearchClawBench.adapter as adapter
    from agent_base.react_agent import default_llm_config

    role_prompt = adapter.Path(adapter.__file__).with_name("role_prompt.md").read_text(encoding="utf-8")
    function_list = [
        name for name in adapter.AVAILABLE_TOOL_MAP
        if name not in adapter.ResearchClawBenchAgent.forbidden_tool_names
    ]
    llm = default_llm_config(model_name="qwen3-235b-a22b-instruct-2507", extra_body={"enable_thinking": False})
    llm["api_base"] = "https://zjuapi.com/v1"
    llm["api_key"] = "DUMMY_DRY_RUN_ONLY"
    agent = adapter.ResearchClawBenchAgent(
        function_list=function_list,
        custom_tools=[triage_evidence],
        llm=llm,
        role_prompt=role_prompt,
        max_rounds=8,
        max_runtime_seconds=60,
    )
    print({
        "constructed": True,
        "model": "qwen3-235b-a22b-instruct-2507",
        "triage_registered": "TriageEvidence" in agent.tool_names,
        "tool_count": len(agent.tool_names),
        "tool_names": agent.tool_names,
    })


if __name__ == "__main__":
    main()
