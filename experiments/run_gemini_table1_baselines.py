#!/usr/bin/env python3
"""Run a fresh Gemini-2.5-Flash-Lite no-thinking baseline matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from gemini_model_config import (
    BASELINE_PROTOCOL_SCHEMA,
    MODEL_ALIAS,
    NATIVE_MODEL,
    RUN_PREFIX,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "experiments/gemini_no_thinking_bridge.py"
LAUNCHER = ROOT / "experiments/run_qwen_table1_baselines.py"
CONTROLLED_PROTOCOL = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"


def parse_ready(process: subprocess.Popen[str], timeout_seconds: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        line = process.stdout.readline() if process.stdout else ""
        if line:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") == "ready" and isinstance(value.get("base_url"), str):
                return value
        if process.poll() is not None:
            break
    raise RuntimeError("Gemini no-thinking bridge did not become ready")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("MBO", "NHO", "SPO", "TMC"), action="append")
    parser.add_argument("--cell", action="append")
    parser.add_argument(
        "--allow-extended-seeds",
        action="store_true",
        help="Pass explicitly selected non-Table-1 seeds to an isolated recheck track.",
    )
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--budget", type=int, default=24)
    # Resolve the workstation Keychain entry by default.  A plaintext file is
    # an explicit opt-in for an isolated legacy run.
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--applicability-domain",
        choices=("none", "mbo_structural_envelope_v2"),
        default="none",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be fresh: {output}")
    if args.api_key_file is not None and not args.api_key_file.is_file():
        raise SystemExit(f"API key file does not exist: {args.api_key_file}")
    if args.parallel < 1 or args.budget < 1:
        raise SystemExit("parallel and budget must be positive")
    if not args.python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {args.python}")
    audit_handle = tempfile.NamedTemporaryFile(prefix="gemini_no_thinking_", suffix=".jsonl", delete=False)
    audit_handle.close()
    bridge_command = [
        str(args.python), str(BRIDGE),
        "--model", NATIVE_MODEL,
        "--port", "0",
        "--audit-log", audit_handle.name,
    ]
    if args.api_key_file is not None:
        bridge_command[2:2] = ["--api-key-file", str(args.api_key_file.resolve())]
    bridge = subprocess.Popen(
        bridge_command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        ready = parse_ready(bridge)
        if ready.get("model") != NATIVE_MODEL:
            raise RuntimeError(
                f"Gemini bridge model mismatch: expected {NATIVE_MODEL}, got {ready.get('model')}"
            )
        command = [
            str(args.python), str(LAUNCHER),
            "--output", str(output),
            "--parallel", str(args.parallel),
            "--budget", str(args.budget),
            "--model", MODEL_ALIAS,
            "--base-url", ready["base_url"],
            # Keep the virtualenv launcher path itself; resolving the symlink
            # would bypass the repository environment.
            "--python", str(args.python.absolute()),
            "--run-prefix", RUN_PREFIX,
            "--protocol-schema", BASELINE_PROTOCOL_SCHEMA,
            "--protocol", str(CONTROLLED_PROTOCOL),
        ]
        if args.api_key_file is not None:
            command[2:2] = ["--api-key-file", str(args.api_key_file.resolve())]
        for task in args.task or []:
            command.extend(["--task", task])
        for cell in args.cell or []:
            command.extend(["--cell", cell])
        if args.allow_extended_seeds:
            command.append("--allow-extended-seeds")
        command.extend(["--applicability-domain", args.applicability_domain])
        completed = subprocess.run(command, cwd=ROOT, text=True)
    finally:
        bridge.terminate()
        try:
            bridge.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bridge.kill()
            bridge.wait(timeout=10)
    if not output.is_dir():
        raise SystemExit("baseline launcher did not create an output directory")
    destination = output / "gemini_no_thinking_bridge.jsonl"
    shutil.copy2(audit_handle.name, destination)
    Path(audit_handle.name).unlink(missing_ok=True)
    rows = read_jsonl(destination)
    verified = [row for row in rows if row.get("no_thinking_verified") is True]
    native_verified = [row for row in rows if row.get("native_model") == NATIVE_MODEL]
    expected_seeds = {
        int(cell.rsplit(":", 1)[1])
        for cell in (args.cell or [])
    }
    if not expected_seeds:
        expected_seeds = {0, 1, 2}
    seeded = [
        row
        for row in rows
        if row.get("seed_forwarded") is True and row.get("seed_requested") in expected_seeds
    ]
    attestation = {
        "schema_version": "gemini_no_thinking_attestation.v1",
        "model_alias": MODEL_ALIAS,
        "native_model": NATIVE_MODEL,
        "transport": "local OpenAI-compatible bridge to New API Gemini native /v1beta",
        "thinking_budget": 0,
        "controlled_protocol": str(CONTROLLED_PROTOCOL.resolve()),
        "controlled_protocol_sha256": hashlib.sha256(CONTROLLED_PROTOCOL.read_bytes()).hexdigest(),
        "temperature": 0.6,
        "max_tokens": 4096,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "enable_thinking": False,
        "context_messages": 10,
        "bridge_log": str(destination.resolve()),
        "bridge_calls": len(rows),
        "verified_no_thinking_calls": len(verified),
        "all_calls_verified_no_thinking": bool(rows) and len(rows) == len(verified),
        "all_calls_verified_native_model": bool(rows) and len(rows) == len(native_verified),
        "expected_seeds": sorted(expected_seeds),
        "seed_forwarded_calls": len(seeded),
        "all_calls_forwarded_selected_seed": bool(rows) and len(rows) == len(seeded),
        "launcher_returncode": completed.returncode,
    }
    (output / "gemini_no_thinking_attestation.json").write_text(
        json.dumps(attestation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    return 0 if (
        completed.returncode == 0
        and attestation["all_calls_verified_no_thinking"]
        and attestation["all_calls_verified_native_model"]
        and attestation["all_calls_forwarded_selected_seed"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
