#!/usr/bin/env python3
"""Collect the two missing AFE arms for the controlled proxy experiment.

The earlier proxy experiment already contains legacy search with and without
the held-out gate.  This script holds the synthetic environment fixed and adds
an AFE portfolio policy with the gate disabled/enabled.  In this one-dimensional
environment, AFE is instantiated by its three frozen roles: local exploitation,
cross-region search, and orthogonal exploration.  The model selects one member
of the portfolio per primary round; the hidden objective is never exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiments.runtime_credentials import read_api_key

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/qwen_required_suite_v3/proxy_factorial"
PLAN = OUT / "proxy_factorial_protocol_v2.json"
PRIOR = ROOT / "experiments/results/qwen_required_suite_v2/proxy_exploitation/results.json"
DECODE = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"
MODEL = "qwen3-235b-a22b-instruct-2507"
SEEDS = (0, 1, 2)
ARMS = ("afe_no_gate", "afe_gate")
ROUNDS = 6
TOLERANCE = 0.25
EXPLOIT_THRESHOLD = 0.72
INITIAL_X = (0.15, 0.45, 0.80)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def true_score(x: float) -> float:
    return 1.2 - 1.5 * (x - 0.32) ** 2


def proxy_score(x: float) -> float:
    base = true_score(x)
    if x >= EXPLOIT_THRESHOLD:
        return base + 1.0 + 0.35 * (x - EXPLOIT_THRESHOLD) / (1.0 - EXPLOIT_THRESHOLD)
    return base


def evaluate(x: float, gate_enabled: bool) -> dict[str, Any]:
    primary = proxy_score(x)
    heldout = true_score(x)
    gap = abs(primary - heldout)
    exploit = x >= EXPLOIT_THRESHOLD
    update_allowed = (gap <= TOLERANCE) if gate_enabled else True
    trusted = primary if update_allowed else heldout
    return {
        "x": x,
        "primary_proxy": primary,
        "heldout_true": heldout,
        "proxy_true_gap": primary - heldout,
        "exploit_region": exploit,
        "update_allowed": update_allowed,
        "trusted_score": trusted,
        "decision": ("admit" if update_allowed else "quarantine_proxy_exploit") if gate_enabled else "source_blind_admit",
        "verification_cost": 1 if gate_enabled else 0,
    }


def parse_selected_x(text: str) -> float | None:
    patterns = (
        r'"selected_x"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'"x"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'\bselected_x\s*=\s*(-?\d+(?:\.\d+)?)',
    )
    for pattern in patterns:
        for raw in re.findall(pattern, text, flags=re.I):
            try:
                value = float(raw)
            except ValueError:
                continue
            if 0.0 <= value <= 1.0:
                return value
    return None


def history_text(history: list[dict[str, Any]], gate_enabled: bool) -> str:
    rows = []
    for index, item in enumerate(history):
        if gate_enabled:
            rows.append(
                f"{index}: x={item['x']:.4f}, primary={item['primary_proxy']:.6f}, "
                f"heldout={item['heldout_true']:.6f}, decision={item['decision']}, "
                f"trusted_score={item['trusted_score']:.6f}"
            )
        else:
            rows.append(f"{index}: x={item['x']:.4f}, observed_score={item['primary_proxy']:.6f}")
    return "\n".join(rows)


def prompt(history: list[dict[str, Any]], gate_enabled: bool) -> str:
    evidence_rule = (
        "Use only admitted/trusted evidence. Quarantined primary observations must not guide optimization."
        if gate_enabled
        else "Use the observed scores as optimization evidence."
    )
    return (
        "You control a scientific optimization loop over x in [0,1]. The objective is hidden. "
        f"{evidence_rule}\n\nHistory:\n{history_text(history, gate_enabled)}\n\n"
        "Construct exactly three new candidate roles using only this history:\n"
        "A local_exploitation: refine the strongest trusted/observed region;\n"
        "B cross_region: test a materially different region with a defensible evidence-based prior;\n"
        "C orthogonal_exploration: sample a sparsely covered region.\n"
        "Then select exactly one of A/B/C for the next primary experiment. Avoid previously tested x values. "
        "Return only compact JSON with numeric fields A, B, C, and selected_x, plus a short rationale."
    )


def fallback_x(history: list[dict[str, Any]], seed: int, round_index: int) -> float:
    grid = (0.05, 0.20, 0.35, 0.50, 0.65, 0.85, 0.95)
    used = [float(row["x"]) for row in history]
    ranked = sorted(
        grid,
        key=lambda x: (min(abs(x - u) for u in used), ((seed + round_index) % len(grid) == grid.index(x))),
        reverse=True,
    )
    return ranked[0]


def run_one(arm: str, seed: int, api_base: str) -> dict[str, Any]:
    gate_enabled = arm == "afe_gate"
    client = OpenAI(api_key=read_api_key(), base_url=api_base, timeout=150.0)
    history = [evaluate(x, gate_enabled) for x in INITIAL_X]
    model_rounds = []
    for round_index in range(ROUNDS):
        text = ""
        x = None
        for _ in range(2):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt(history, gate_enabled)}],
                temperature=0.6,
                max_tokens=512,
                seed=seed * 100 + round_index,
            )
            text = response.choices[0].message.content or ""
            x = parse_selected_x(text)
            if x is not None:
                break
        if x is None:
            x = fallback_x(history, seed, round_index)
        used = {round(float(item["x"]), 6) for item in history}
        if round(x, 6) in used:
            x = fallback_x(history, seed, round_index)
        observation = evaluate(float(x), gate_enabled)
        observation.update({"round": round_index, "model_response": text})
        history.append(observation)
        model_rounds.append(observation)
    oracle = true_score(0.32)
    metrics = {
        "proxy_best": max(float(item["primary_proxy"]) for item in history),
        "true_best": max(float(item["heldout_true"]) for item in history),
        "model_round_true_mean": statistics.mean(float(item["heldout_true"]) for item in model_rounds),
        "model_round_proxy_mean": statistics.mean(float(item["primary_proxy"]) for item in model_rounds),
        "exploit_rate": statistics.mean(float(item["exploit_region"]) for item in model_rounds),
        "false_admissions": sum(bool(item["exploit_region"] and item["update_allowed"]) for item in model_rounds),
        "quarantined_exploits": sum(bool(item["exploit_region"] and not item["update_allowed"]) for item in model_rounds),
        "verification_cost": sum(int(item["verification_cost"]) for item in model_rounds),
        "cumulative_true_regret": sum(oracle - float(item["heldout_true"]) for item in model_rounds),
    }
    return {"arm": arm, "seed": seed, "history": history, "model_rounds": model_rounds, "metrics": metrics}


def expected_plan() -> dict[str, Any]:
    decode = json.loads(DECODE.read_text(encoding="utf-8"))
    return {
        "schema_version": "qwen_proxy_factorial.v2",
        "objective": "complete the missing AFE-search arms in the controlled proxy 2x2 factorial",
        "model": MODEL,
        "api_base": decode["base_url"],
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "new_arms": list(ARMS),
        "full_factorial_arms": ["legacy_no_gate", "legacy_gate", "afe_no_gate", "afe_gate"],
        "initial_x": list(INITIAL_X),
        "exploit_threshold": EXPLOIT_THRESHOLD,
        "triage_tolerance": TOLERANCE,
        "temperature": 0.6,
        "max_tokens": 512,
        "afe_roles": ["local_exploitation", "cross_region", "orthogonal_exploration"],
        "prior_legacy_results": str(PRIOR.resolve()),
        "prior_legacy_results_sha256": sha256(PRIOR),
        "decode_protocol": str(DECODE.resolve()),
        "decode_protocol_sha256": sha256(DECODE),
        "selection_rule": "retain all arm-seed runs; no score-based retry, seed replacement, or run deletion",
        "claim_boundary": "controlled one-dimensional instantiation of the AFE portfolio roles and held-out gate; not a claim that the complete chemistry-specific AFE implementation transfers unchanged to this synthetic domain",
    }


def prepare() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = expected_plan()
    if not PLAN.exists():
        PLAN.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    observed = json.loads(PLAN.read_text(encoding="utf-8"))
    if observed != expected:
        raise SystemExit(f"frozen plan mismatch: {PLAN}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    plan = prepare()
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "new_runs": len(ARMS) * len(SEEDS), "plan": str(PLAN)}, indent=2))
        return 0
    result_path = OUT / "afe_runs.json"
    if result_path.exists():
        raise SystemExit(f"result already exists: {result_path}")
    runs = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(run_one, arm, seed, plan["api_base"]): (arm, seed) for arm in ARMS for seed in SEEDS}
        for future in as_completed(futures):
            row = future.result()
            runs.append(row)
            print(json.dumps({"arm": row["arm"], "seed": row["seed"], **row["metrics"]}, ensure_ascii=False), flush=True)
    runs.sort(key=lambda row: (row["arm"], row["seed"]))
    payload = {
        "schema_version": "qwen_proxy_factorial_afe_runs.v2",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "runs": runs,
        "plan": str(PLAN.resolve()),
        "plan_sha256": sha256(PLAN),
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
