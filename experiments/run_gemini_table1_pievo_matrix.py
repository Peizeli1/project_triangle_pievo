#!/usr/bin/env python3
"""Launch fresh Gemini-2.5-Flash-Lite no-thinking PiEvo/RADP Table 1 cells."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gemini_model_config import (
    MODEL_ALIAS,
    NATIVE_MODEL,
    PIEVO_PROTOCOL_SCHEMA,
    RUN_PREFIX,
    TABLE1_TRACK,
)


ROOT = Path(__file__).resolve().parents[1]
CELL_RUNNER = ROOT / "experiments/run_gemini_table1_pievo_cell.py"
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = ("pievo", "radp")
SEEDS = (0, 1, 2)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_cell(value: str) -> tuple[str, str, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("cell must be METHOD:TASK:SEED")
    method, task, seed_text = parts
    if method not in METHODS or task not in TASKS:
        raise argparse.ArgumentTypeError(f"unsupported cell: {value}")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed in cell: {value}") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError(f"seed must be non-negative: {value}")
    return method, task, seed


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def run_cell(spec: dict[str, Any]) -> dict[str, Any]:
    command = [
        spec["python"],
        str(CELL_RUNNER),
        "--task", spec["task"],
        "--method", spec["method"],
        "--seed", str(spec["seed"]),
        "--budget", "24",
        "--output", str(spec["output"]),
        "--python", spec["python"],
        "--pievo-generator", spec["pievo_generator"],
        "--candidate-recovery", spec["candidate_recovery"],
        "--run-track", spec["run_track"],
        "--applicability-domain", spec["applicability_domain"],
    ]
    if spec["api_key_file"] is not None:
        command[command.index("--python"):command.index("--python")] = [
            "--api-key-file", str(spec["api_key_file"]),
        ]
    if spec["allow_extended_seed"]:
        command.append("--allow-extended-seed")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    manifest = load_json(spec["output"] / "run_manifest.json")
    return {
        "key": spec["key"],
        "task": spec["task"],
        "method": spec["method"],
        "seed": spec["seed"],
        "output": str(spec["output"]),
        "returncode": completed.returncode,
        "manifest": manifest,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--cell", type=parse_cell, action="append")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Explicit seed set. Seeds outside 0, 1, 2 require --allow-extended-seeds.",
    )
    parser.add_argument("--allow-extended-seeds", action="store_true")
    parser.add_argument("--pievo-generator", choices=("native", "common"), default="native")
    parser.add_argument(
        "--candidate-recovery",
        choices=("none", "outcome_blind_structural_recovery_v2"),
        default="none",
    )
    parser.add_argument("--run-track", default=TABLE1_TRACK)
    parser.add_argument(
        "--applicability-domain",
        choices=("none", "mbo_structural_envelope_v2"),
        default="none",
    )
    parser.add_argument("--parallel", type=int, default=2)
    # Resolve the workstation Keychain entry by default.  A plaintext file is
    # an explicit opt-in for an isolated legacy run.
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    args = parser.parse_args()
    if args.parallel < 1:
        raise SystemExit("parallel must be positive")
    if args.api_key_file is not None and not args.api_key_file.is_file():
        raise SystemExit(f"API key file does not exist: {args.api_key_file}")
    if not args.python.is_file():
        raise SystemExit("missing Python interpreter")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if any(seed < 0 for seed in args.seeds):
        raise SystemExit("seeds must be non-negative")
    if args.cell:
        selected = list(dict.fromkeys(args.cell))
    else:
        selected_tasks = tuple(args.task) if args.task else TASKS
        selected = [
            (method, task, seed)
            for task in selected_tasks
            for method in METHODS
            for seed in args.seeds
        ]
    if not args.allow_extended_seeds and any(seed not in SEEDS for _, _, seed in selected):
        raise SystemExit("extended seeds require --allow-extended-seeds")
    selected_tasks = [task for task in TASKS if any(value[1] == task for value in selected)]
    run_map: dict[str, str] = {}
    specs: list[dict[str, Any]] = []
    for method, task, seed in selected:
        key = f"{method}:{task}:{seed}"
        cell_output = output / f"{RUN_PREFIX}_{method}_{task.lower()}_seed{seed}_formal24"
        run_map[key] = str(cell_output.resolve())
        specs.append(
            {
                "key": key,
                "method": method,
                "task": task,
                "seed": seed,
                "output": cell_output,
                "api_key_file": args.api_key_file.resolve() if args.api_key_file is not None else None,
                "python": str(args.python.absolute()),
                "pievo_generator": args.pievo_generator,
                "candidate_recovery": args.candidate_recovery,
                "run_track": args.run_track,
                "applicability_domain": args.applicability_domain,
                "allow_extended_seed": args.allow_extended_seeds and seed not in SEEDS,
            }
        )
    write_json(output / "matched_run_map.json", run_map)
    protocol = {
        "schema_version": PIEVO_PROTOCOL_SCHEMA,
        "model_alias": MODEL_ALIAS,
        "native_model": NATIVE_MODEL,
        "transport": "local OpenAI-compatible bridge to Gemini native /v1beta",
        "thinking_budget": 0,
        "seed_forwarding_required": True,
        "tasks": selected_tasks,
        "methods": list(METHODS),
        "seeds": sorted({seed for _, _, seed in selected}),
        "selected_cells": [spec["key"] for spec in specs],
        "primary_budget_per_cell": 24,
        "parallel_workers": args.parallel,
        "pievo_generator": args.pievo_generator,
        "candidate_recovery": args.candidate_recovery,
        "run_track": args.run_track,
        "applicability_domain": args.applicability_domain,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Fresh Gemini no-thinking matched released-evaluator reproduction; not recovery of unpublished hidden paper assets.",
    }
    write_json(output / "matched_protocol.json", protocol)
    results: list[dict[str, Any]] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_cell, spec): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            manifest = result["manifest"]
            audit = manifest.get("audit", {}) if isinstance(manifest, dict) else {}
            runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
            ok = bool(
                result["returncode"] == 0
                and manifest.get("status") == "completed"
                and audit.get("primary_complete") is True
                and runtime.get("all_bridge_calls_no_thinking") is True
                and runtime.get("all_bridge_calls_native_model") is True
                and runtime.get("all_bridge_calls_forwarded_seed") is True
            )
            if not ok:
                failed += 1
            print(
                json.dumps(
                    {
                        "event": "cell_complete",
                        "key": result["key"],
                        "ok": ok,
                        "primary": audit.get("primary"),
                        "bridge_no_thinking": runtime.get("all_bridge_calls_no_thinking"),
                        "seed_forwarded": runtime.get("all_bridge_calls_forwarded_seed"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    results.sort(key=lambda item: item["key"])
    write_json(output / "cell_status.json", results)
    protocol["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["completed_cells"] = len(results) - failed
    protocol["failed_cells"] = failed
    write_json(output / "matched_protocol.json", protocol)
    print(json.dumps({"event": "matrix_complete", "output": str(output), "completed_cells": len(results) - failed, "failed_cells": failed}, ensure_ascii=False), flush=True)
    return 0 if failed == 0 and len(results) == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
