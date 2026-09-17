#!/usr/bin/env python3
"""Generate one SciAgentArena solution through a concrete agent harness.

The helper writes one JSON record to --out.  It never invokes the benchmark
scorer and is therefore safe to run in the generation boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DD_ROOT = ROOT / "official_sources/SciAgentArena/evaluations/dd"
MODEL = "qwen3-235b-a22b-instruct-2507"
API_BASE = "https://zjuapi.com/v1"
TEMPERATURE = 0.6
MAX_OUTPUT_TOKENS = 8192

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DD_ROOT))

from experiments.runtime_credentials import read_api_key
from agents.prompts import build_system_instruction, build_user_prompt


def strip_code(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    if text.startswith("python\n"):
        return text[7:].strip()
    return text


def task_prompts(task_path: Path) -> tuple[str, str]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    system = build_system_instruction(task)
    user = build_user_prompt(task)
    return system, user


def direct_openai_call(system: str, user: str, seed: int) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=read_api_key(), base_url=API_BASE, timeout=240.0)
    started = time.perf_counter()
    error = None
    raw = ""
    usage: dict[str, Any] = {}
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            seed=seed,
        )
        raw = response.choices[0].message.content or ""
        if response.usage is not None:
            usage = response.usage.model_dump()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "code": strip_code(raw), "raw": raw, "returncode": 1 if error else 0,
        "runtime_seconds": time.perf_counter() - started, "usage": usage,
        "error": error,
    }


def claude_call(system: str, user: str, seed: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = read_api_key()
    env["ANTHROPIC_BASE_URL"] = "https://zjuapi.com"
    cmd = [
        "claude", "--bare", "--print", "--model", MODEL,
        "--tools", "", "--output-format", "json",
        "--no-session-persistence", "--max-budget-usd", "1.00",
        "--system-prompt", system,
        user + f"\n\nReproducibility seed: {seed}. Return only the Python program.",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=DD_ROOT, env=env, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - started
    payload = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
    raw = str(payload.get("result") or proc.stdout)
    usage = payload.get("usage") or {}
    return {
        "code": strip_code(raw), "raw": raw, "returncode": proc.returncode,
        "runtime_seconds": elapsed, "usage": usage,
        "model_usage": payload.get("modelUsage") or {},
        "stderr_tail": proc.stderr[-2000:],
    }


def codex_call(system: str, user: str, seed: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["ZJU_TASK_API_KEY"] = read_api_key()
    prompt = system + "\n\n" + user + f"\n\nReproducibility seed: {seed}. Return only the Python program."
    with tempfile.TemporaryDirectory(prefix="saa-codex-") as tmp:
        tmp_path = Path(tmp)
        # Codex is an interactive coding harness and reasonably tries to inspect
        # referenced inputs. Give it only public task inputs, never scorers or
        # answer keys, and ask it to commit the program to a file.
        for relative in ("tasks_batch/data", "tasks_interactive/data"):
            source = DD_ROOT / relative
            if source.is_dir():
                shutil.copytree(source, tmp_path / relative)
        out = Path(tmp) / "last.txt"
        solution = Path(tmp) / "solution.py"
        prompt += (
            "\n\nYou may inspect the task input files in this isolated workspace. "
            "Write the final complete program to solution.py. Do not merely describe it."
        )
        cmd = [
            "codex", "exec", "--ignore-user-config", "--skip-git-repo-check",
            "--ephemeral", "-s", "workspace-write", "-m", MODEL,
            "-c", 'model_provider="zju"',
            "-c", 'model_providers.zju.name="ZJU"',
            "-c", 'model_providers.zju.base_url="https://zjuapi.com/v1"',
            "-c", 'model_providers.zju.env_key="ZJU_TASK_API_KEY"',
            "-c", 'model_providers.zju.wire_api="responses"',
            "-o", str(out), prompt,
        ]
        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=360)
        elapsed = time.perf_counter() - started
        raw = out.read_text(encoding="utf-8") if out.is_file() else proc.stdout
        code = solution.read_text(encoding="utf-8") if solution.is_file() else strip_code(raw)
    token_match = re.findall(r"tokens used\s*\n\s*([0-9,]+)", proc.stderr)
    tokens = int(token_match[-1].replace(",", "")) if token_match else None
    return {
        "code": code, "raw": raw, "returncode": proc.returncode,
        "runtime_seconds": elapsed, "usage": {"total_tokens": tokens},
        "stderr_tail": proc.stderr[-2000:],
    }


def ai_researcher_call(system: str, user: str, seed: int) -> dict[str, Any]:
    ai_root = ROOT / "official_sources/AI-Researcher"
    sys.path.insert(0, str(ai_root))
    os.environ.setdefault("DEFAULT_LOG", "0")
    os.environ.setdefault("DEBUG", "0")
    os.environ["API_BASE_URL"] = API_BASE
    os.environ["OPENAI_API_KEY"] = read_api_key()
    from research_agent.inno import MetaChain
    from research_agent.inno.types import Agent
    import research_agent.inno.core as core

    calls: list[dict[str, Any]] = []
    original = core.completion

    def recorded(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("temperature", TEMPERATURE)
        kwargs.setdefault("max_tokens", MAX_OUTPUT_TOKENS)
        response = original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        calls.append(usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {}))
        return response

    core.completion = recorded
    started = time.perf_counter()
    error = None
    raw = ""
    try:
        agent = Agent(name="SciAgentArena", model="openai/" + MODEL, instructions=system)
        response = MetaChain().run(
            agent,
            [{"role": "user", "content": user + f"\n\nReproducibility seed: {seed}. Return only the Python program."}],
            debug=False,
            max_turns=1,
        )
        raw = str(response.messages[-1].get("content") or "")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        core.completion = original
    return {
        "code": strip_code(raw), "raw": raw, "returncode": 1 if error else 0,
        "runtime_seconds": time.perf_counter() - started, "usage_calls": calls,
        "usage": {"total_tokens": sum(int(c.get("total_tokens") or 0) for c in calls)},
        "error": error,
    }


def researchharness_call(system: str, user: str, seed: int) -> dict[str, Any]:
    from researchharness import run_agent

    trace_dir = Path(os.environ.get("SAA_TRACE_DIR", "/tmp/saa-rh-trace"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    error = None
    raw = ""
    try:
        raw = run_agent(
            user + f"\n\nReproducibility seed: {seed}. Return only the Python program.",
            workspace_root=str(DD_ROOT), model_name=MODEL,
            api_key=read_api_key(), api_base=API_BASE,
            role_prompt=system, tools=[], require_env=False,
            temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS,
            max_input_tokens=32768, max_rounds=2, max_runtime_seconds=300,
            omit_generate_params=["presence_penalty"], trace_dir=str(trace_dir),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "code": strip_code(str(raw)), "raw": str(raw),
        "returncode": 1 if error else 0,
        "runtime_seconds": time.perf_counter() - started,
        "usage": {}, "error": error,
    }


CALLERS: dict[str, Callable[[str, str, int], dict[str, Any]]] = {
    "direct_openai": direct_openai_call,
    "claude_code": claude_call,
    "codex_cli": codex_call,
    "ai_researcher": ai_researcher_call,
    "researchharness": researchharness_call,
}


def _sum_tokens(call: dict[str, Any]) -> int:
    usage = call.get("usage") or {}
    if usage.get("total_tokens") is not None:
        return int(usage["total_tokens"])
    total = 0
    for item in (call.get("model_usage") or {}).values():
        total += int(item.get("inputTokens") or 0)
        total += int(item.get("outputTokens") or 0)
        total += int(item.get("cacheReadInputTokens") or 0)
        total += int(item.get("cacheCreationInputTokens") or 0)
    return total


def generate_variant(
    caller: Callable[[str, str, int], dict[str, Any]],
    system: str,
    user: str,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    def invoke(call_system: str, call_user: str, offset: int) -> dict[str, Any]:
        result = caller(call_system, call_user, seed + offset)
        calls.append(result)
        return result

    primary = invoke(system, user, 0)
    final = primary
    if variant == "self_review" and primary.get("code"):
        review_user = (
            user
            + "\n\nHere is an initial proposed program:\n```python\n"
            + primary["code"]
            + "\n```\nIndependently check it against every task requirement and return a corrected, complete Python program only."
        )
        final = invoke(system, review_user, 1009)
    elif variant == "triage" and primary.get("code"):
        independent = invoke(
            system,
            user + "\n\nProduce an independent solution. Do not assume another candidate is correct. Return only Python.",
            2017,
        )
        adjudication = (
            user
            + "\n\nTwo independently produced candidates follow. Treat them as evidence, not as truth. "
            "Check provenance, task constraints, data paths, output schema, invalid-input handling, and unsupported assumptions. "
            "Admit a shared choice only when independently supported; otherwise quarantine the disputed step and synthesize a conservative executable correction. "
            "Do not use or infer hidden benchmark answers. Return one complete Python program only.\n\n"
            + "PRIMARY:\n```python\n" + primary.get("code", "") + "\n```\n\n"
            + "INDEPENDENT:\n```python\n" + independent.get("code", "") + "\n```"
        )
        final = invoke(system, adjudication, 3011)
    elif variant.startswith("triage_pievo") and primary.get("code"):
        # ── generate independent candidates (k controls count) ───────
        expansion = invoke(
            system,
            user + "\n\nAct as the frontier-expansion role: seek a materially different valid strategy and return only Python.",
            2017,
        )
        candidates = [("EXPLOITATION", primary), ("EXPANSION", expansion)]
        if variant != "triage_pievo_k1":
            probe = invoke(
                system,
                user + "\n\nAct as the mechanism-probe role: target likely failure assumptions and return a diagnostic but complete solution in Python only.",
                3011,
            )
            candidates.append(("MECHANISM PROBE", probe))
        if variant == "triage_pievo_k3":
            extra = invoke(
                system,
                user + "\n\nAct as an independent verifier: produce a fresh solution validating all constraints from scratch. Return only Python.",
                5003,
            )
            candidates.append(("INDEPENDENT VERIFICATION", extra))
        # ── adjudication (ablation-specific prompt) ──────────────────
        candidate_block = "\n\n".join(
            f"{label}:\n```python\n{c.get('code', '')}\n```"
            for label, c in candidates
        )
        if variant == "triage_pievo_no_holdout":
            admission = (
                "Merge the candidates below into one optimal program. "
                "Accept the strongest approach without requiring independent "
                "corroboration for individual steps. "
                "Do not access or infer hidden scorer answers. "
                "Return one complete executable Python program only."
            )
        elif variant == "triage_pievo_no_channel":
            admission = (
                "Apply evidence admission to the candidates below. "
                "A proposal is not trustworthy merely because it is confident. "
                "Check agreement on task requirements, paths, schema, validity "
                "handling, and failure assumptions. "
                "Preserve supported improvements, quarantine unsupported steps, "
                "and backtrack to a safer alternative when necessary. "
                "Do not access or infer hidden scorer answers. "
                "Return one complete executable Python program only."
            )
        elif variant == "triage_pievo_no_backtrack":
            admission = (
                "Apply evidence admission to the candidates below. "
                "A proposal is not trustworthy merely because it is confident. "
                "Check agreement on task requirements, provenance, paths, schema, "
                "validity handling, and failure assumptions. "
                "Preserve supported improvements and accept the best available "
                "candidate without reverting to simpler alternatives. "
                "Do not access or infer hidden scorer answers. "
                "Return one complete executable Python program only."
            )
        elif variant == "triage_pievo_same_source":
            admission = (
                "Apply evidence admission to the candidates below. "
                "If multiple candidates converge on the same step, treat that "
                "convergence as independent confirmation regardless of shared "
                "strategy or source. "
                "Check agreement on task requirements, provenance, paths, schema, "
                "validity handling, and failure assumptions. "
                "Preserve supported improvements, quarantine unsupported steps, "
                "and backtrack to a safer alternative when necessary. "
                "Do not access or infer hidden scorer answers. "
                "Return one complete executable Python program only."
            )
        else:
            # triage_pievo (Full, k=2), triage_pievo_k1, triage_pievo_k3
            if variant == "triage_pievo_k1":
                cand_label = "the exploitation and expansion candidates"
            elif variant == "triage_pievo_k3":
                cand_label = "the four candidates"
            else:
                cand_label = "the exploitation, expansion, and mechanism-probe candidates"
            admission = (
                f"Apply evidence admission to {cand_label} below. "
                "A proposal is not trustworthy merely because it is confident. "
                "Check agreement on task requirements, provenance, paths, schema, "
                "validity handling, and failure assumptions. "
                "Preserve supported improvements, quarantine unsupported steps, "
                "and backtrack to a safer alternative when necessary. "
                "Do not access or infer hidden scorer answers. "
                "Return one complete executable Python program only."
            )
        adjudication = user + "\n\n" + admission + "\n\n" + candidate_block
        final = invoke(system, adjudication, 6007 if variant == "triage_pievo_k3" else 4001)

    return {
        "code": final.get("code", ""),
        "raw": final.get("raw", ""),
        "returncode": final.get("returncode", 1),
        "runtime_seconds": sum(float(c.get("runtime_seconds") or 0.0) for c in calls),
        "usage": {"total_tokens": sum(_sum_tokens(c) for c in calls)},
        "error": final.get("error"),
        "calls": calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", choices=sorted(CALLERS), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "base", "self_review", "triage", "triage_pievo",
            "triage_pievo_k1", "triage_pievo_k3",
            "triage_pievo_no_holdout", "triage_pievo_no_channel",
            "triage_pievo_no_backtrack", "triage_pievo_same_source",
        ),
        default="base",
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    system, user = task_prompts(args.task.resolve())
    result = generate_variant(CALLERS[args.harness], system, user, args.seed, args.variant)
    payload = {
        "harness": args.harness, "variant": args.variant, "model": MODEL, "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "seed": args.seed,
        "task_path": str(args.task.resolve()), **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if result["returncode"] == 0 and bool(result["code"].strip()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
