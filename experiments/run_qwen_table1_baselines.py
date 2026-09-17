#!/usr/bin/env python3
"""Launch the missing Qwen Table 1 baseline cells with bounded concurrency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/baseline_table1_runner.py"
CONTROLLED_PROTOCOL = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
SEEDS = (0, 1, 2)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_cell(value: str) -> tuple[str, str, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("cell must be METHOD:TASK:SEED")
    method, task, seed_text = parts
    if method not in METHODS:
        raise argparse.ArgumentTypeError(f"unsupported method in cell: {method}")
    if task not in TASKS:
        raise argparse.ArgumentTypeError(f"unsupported task in cell: {task}")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed in cell: {seed_text}") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError(f"seed must be non-negative: {seed}")
    return method, task, seed


def run_cell(spec: dict[str, Any]) -> dict[str, Any]:
    command = [
        spec["python"],
        str(RUNNER),
        "--task",
        spec["task"],
        "--method",
        spec["method"],
        "--seed",
        str(spec["seed"]),
        "--budget",
        str(spec["budget"]),
        "--model",
        spec["model"],
        "--base-url",
        spec["base_url"],
        "--output",
        str(spec["output"]),
        "--applicability-domain",
        spec["applicability_domain"],
        "--protocol",
        str(spec["protocol"]),
    ]
    if spec["api_key_file"] is not None:
        command[command.index("--output"):command.index("--output")] = [
            "--api-key-file",
            str(spec["api_key_file"]),
        ]
    environment = os.environ.copy()
    # The child reads the key from the file only. Do not propagate a raw key
    # that may have been exported by an unrelated shell session.
    environment.pop("ZJU_TASK_API_KEY", None)
    environment.pop("OPENAI_API_KEY", None)
    process = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    summary_path = spec["output"] / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = {"status": "failed", "error": "summary.json missing or invalid"}
    return {
        "key": spec["key"],
        "task": spec["task"],
        "method": spec["method"],
        "seed": spec["seed"],
        "output": str(spec["output"]),
        "returncode": process.returncode,
        "summary": summary,
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-run-map", type=Path)
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument(
        "--cell",
        type=parse_cell,
        action="append",
        help="run only an exact METHOD:TASK:SEED cell; repeat as needed",
    )
    parser.add_argument(
        "--allow-extended-seeds",
        action="store_true",
        help="Allow explicit non-Table-1 seeds for an isolated recheck track.",
    )
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--base-url", default="https://zjuapi.com/v1")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Optional plaintext key file; omitted uses the configured local Keychain helper.",
    )
    parser.add_argument("--python", type=Path, help="child Python executable; defaults to the repository .venv")
    parser.add_argument("--run-prefix", default="qwen")
    parser.add_argument("--protocol-schema", default="qwen_table1_baseline_protocol.v1")
    parser.add_argument("--protocol", type=Path, default=CONTROLLED_PROTOCOL)
    parser.add_argument(
        "--applicability-domain",
        choices=("none", "mbo_structural_envelope_v2"),
        default="none",
    )
    args = parser.parse_args()
    if args.parallel < 1 or args.budget < 1:
        raise SystemExit("parallel and budget must be positive")
    if args.api_key_file is not None and not args.api_key_file.is_file():
        raise SystemExit(f"API key file does not exist: {args.api_key_file}")
    if not args.protocol.is_file():
        raise SystemExit(f"controlled protocol does not exist: {args.protocol}")
    if args.matched_run_map is not None and not args.matched_run_map.is_file():
        raise SystemExit(f"matched run map does not exist: {args.matched_run_map}")
    # Keep the venv launcher path itself. Resolving its symlink would bypass
    # the virtual environment and run the Homebrew base interpreter.
    child_python = (args.python or (ROOT / ".venv/bin/python")).absolute()
    if not child_python.is_file():
        child_python = Path(sys.executable).absolute()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.cell:
        selected_cells = list(dict.fromkeys(args.cell))
    else:
        selected_tasks = tuple(args.task) if args.task else TASKS
        selected_cells = [(method, task, seed) for task in selected_tasks for method in METHODS for seed in SEEDS]
    if not args.allow_extended_seeds and any(seed not in SEEDS for _, _, seed in selected_cells):
        raise SystemExit("extended seeds require --allow-extended-seeds")
    specs: list[dict[str, Any]] = []
    run_map: dict[str, str] = {}
    for method, task, seed in selected_cells:
        key = f"{method}:{task}:{seed}"
        cell_output = output / f"{args.run_prefix}_{method}_{task.lower()}_seed{seed}_formal24"
        run_map[key] = str(cell_output.resolve())
        specs.append(
            {
                "key": key,
                "task": task,
                "method": method,
                "seed": seed,
                "budget": args.budget,
                "model": args.model,
                "base_url": args.base_url,
                "api_key_file": args.api_key_file.resolve() if args.api_key_file is not None else None,
                "python": str(child_python),
                "output": cell_output,
                "applicability_domain": args.applicability_domain,
                "protocol": args.protocol.resolve(),
            }
        )
    if not specs:
        raise SystemExit("no cells selected")
    selected_tasks = tuple(task for task in TASKS if any(item[1] == task for item in selected_cells))
    selected_methods = tuple(method for method in METHODS if any(item[0] == method for item in selected_cells))
    write_json(output / "baseline_run_map.json", run_map)
    write_json(
        output / "baseline_protocol.json",
        {
            "schema_version": args.protocol_schema,
            "run_prefix": args.run_prefix,
            "model": args.model,
            "base_url": args.base_url,
            "tasks": list(selected_tasks),
            "methods": list(selected_methods),
            "seeds": sorted({seed for _, _, seed in selected_cells}),
            "selected_cells": [spec["key"] for spec in specs],
            "primary_budget_per_cell": args.budget,
            "parallel_workers": args.parallel,
            "python": str(child_python),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "claim_boundary": "Matched released evaluator reproduction; not exact hidden paper leaderboard recovery.",
            "applicability_domain_contract": args.applicability_domain,
            "controlled_protocol": str(args.protocol.resolve()),
            "controlled_protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        },
    )
    results: list[dict[str, Any]] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_cell, spec): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            summary = result["summary"]
            ok = result["returncode"] == 0 and summary.get("status") == "completed"
            if not ok:
                failed += 1
            print(
                json.dumps(
                    {
                        "event": "cell_complete",
                        "key": result["key"],
                        "ok": ok,
                        "primary": summary.get("primary_count"),
                        "best": summary.get("best_output"),
                        "sq_percent": summary.get("sq_percent"),
                        "attempts": summary.get("attempts"),
                        "error": summary.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    results.sort(key=lambda item: item["key"])
    write_json(output / "cell_status.json", results)
    protocol = json.loads((output / "baseline_protocol.json").read_text(encoding="utf-8"))
    protocol["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["completed_cells"] = len(results) - failed
    protocol["failed_cells"] = failed
    write_json(output / "baseline_protocol.json", protocol)
    print(json.dumps({"event": "baseline_matrix_complete", "output": str(output), "completed_cells": len(results) - failed, "failed_cells": failed}, ensure_ascii=False), flush=True)
    return 0 if failed == 0 and len(results) == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
