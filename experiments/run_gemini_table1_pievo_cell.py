#!/usr/bin/env python3
"""Run one PiEvo or RADP Gemini-2.5-Flash-Lite no-thinking Table 1 cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gemini_model_config import (
    MODEL_ALIAS,
    NATIVE_MODEL,
    RUN_MANIFEST_SCHEMA,
    SMOKE_TRACK,
    TABLE1_TRACK,
)

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "experiments/gemini_no_thinking_bridge.py"
CONTROLLED_PROTOCOL = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"
TASK_CONFIGS = {
    "MBO": ROOT / "experiments/configs/official_pievo_mbo_task.yaml",
    "NHO": ROOT / "experiments/configs/official_pievo_nho_task.yaml",
    "SPO": ROOT / "experiments/configs/official_pievo_spo_task.yaml",
    "TMC": ROOT / "experiments/configs/official_pievo_tmc_task.yaml",
}
MODEL_CONFIGS = {
    "MBO": ROOT / "experiments/configs/official_pievo_mbo_model_short_smiles.yaml",
    "NHO": ROOT / "experiments/configs/official_pievo_nho_model.yaml",
    "SPO": ROOT / "experiments/configs/official_pievo_spo_model.yaml",
    "TMC": ROOT / "experiments/configs/official_pievo_tmc_model.yaml",
}
REPOS = {"pievo": ROOT / "official_sources/PiEvo_matched_qwen235b", "radp": ROOT / "official_sources/PiEvo_radp_qwen235b"}
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}
EMBEDDING_MODEL = "/Users/michael/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


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


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decode_protocol(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"controlled decode protocol is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"controlled decode protocol must be a JSON object: {path}")
    required = (
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "presence_penalty",
        "repetition_penalty",
        "enable_thinking",
        "context_messages",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise RuntimeError(f"controlled decode protocol missing fields: {', '.join(missing)}")
    if value["enable_thinking"] is not False:
        raise RuntimeError("controlled Gemini protocol must explicitly disable thinking")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_gemini_task_config(source: Path, destination: Path, bridge_url: str) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        'UTIL_LLM_CONFIG_BASE_URL: "https://zjuapi.com/v1"',
        f'UTIL_LLM_CONFIG_BASE_URL: "{bridge_url}"',
    )
    text = text.replace(
        'UTIL_LLM_CONFIG_NAME: "qwen3-235b-a22b-instruct-2507"',
        f'UTIL_LLM_CONFIG_NAME: "{MODEL_ALIAS}"',
    )
    destination.write_text(text, encoding="utf-8")


def configure_environment(
    task: str,
    method: str,
    seed: int,
    budget: int,
    output: Path,
    repo: Path,
    bridge_url: str,
    *,
    smoke: bool,
    pievo_generator: str,
    recovery_contract: str,
    run_track: str,
    applicability_domain: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("ZJU_TASK_API_KEY", "UTIL_LLM_CONFIG_API_KEY"):
        environment[name] = "local-bridge"
    environment["OPENAI_API_KEY"] = "local-bridge"
    environment.update(
        {
            "PIEVO_TASK": task,
            "PIEVO_METHOD": method,
            "PIEVO_SEED": str(seed),
            "PIEVO_RUN_STAGE": "smoke" if smoke else "formal",
            "PIEVO_RUN_TRACK": SMOKE_TRACK if smoke else run_track,
            "PIEVO_MAX_PRIMARY_EVALS": str(budget),
            "PIEVO_MAX_INVALID_ATTEMPTS": "8",
            "PIEVO_MAX_TOTAL_TOKENS": "350000",
            "PIEVO_MAX_RETRIES": "0",
            "PIEVO_STOP_REQUESTED": "0",
            "PIEVO_DISABLE_PLOTS": "1",
            "PIEVO_EMBEDDING_OFFLINE": "1",
            "PIEVO_EMBEDDING_MODEL_PATH": EMBEDDING_MODEL,
            "PIEVO_SURROGATE_ROOT": str(repo / "surrogates/envs"),
            "PIEVO_CHEMBL_MODEL_PATH": str(repo / "surrogates/envs/ChEMBL35/models/best_r2_model.pt"),
            "PIEVO_EXPERIMENT_ROOT": str(ROOT),
            "PIEVO_AGENT_BASE_URL": bridge_url,
            "PIEVO_AGENT_MODEL": MODEL_ALIAS,
            "PIEVO_AGENT_TEMPERATURE": "0.6",
            "PIEVO_AGENT_MAX_TOKENS": "4096",
            "PIEVO_DECODE_PROTOCOL": str(CONTROLLED_PROTOCOL),
            "PIEVO_AGENT_API_KEY": "local-bridge",
            "PIEVO_GROUP_BUDGET": str(budget + 16),
            "PIEVO_MAX_TURN": "280",
            "PIEVO_WARMUP_ROUNDS": "5",
            "PIEVO_ANOMALY_THRESHOLD": "0.85",
            "PIEVO_ANOMALY_COUNT_THRESHOLD": "3",
            "PIEVO_OFFICIAL_ASSETS_VERIFIED": "0",
            "PIEVO_MATCHED_EVALUATOR_ASSETS_VERIFIED": "1",
            "PYTHONHASHSEED": str(seed),
            "WORKING_RESULT_DIR": str(output),
            "PIEVO_OFFICIAL_PIFLOW_REPO": str(ROOT / "official_sources/PiFlow"),
            "PYTHONPATH": os.pathsep.join([str(repo / "surrogates/envs"), str(repo), str(ROOT), environment.get("PYTHONPATH", "")]),
        }
    )
    if task == "MBO":
        environment.update(
            {
                "PIEVO_MBO_FORBIDDEN_CANDIDATES": "C1CCCCCCC2C1CCCCCCC3C2CCCCCCC3",
                "PIEVO_AFE_POOL_PATH": str(ROOT / "experiments/results/official_chembl35_pool_seed0/chembl35_pool_seed0.jsonl"),
                "PIEVO_AFE_POOL_LIMIT": "50000",
                "PIEVO_AFE_POOL_PROPOSALS": "256",
            }
        )
    elif task == "SPO":
        environment.update(
            {
                "PIEVO_AFE_POOL_PATH": str(ROOT / "official_sources/PiFlow/envs/AgenX_Supercon/data/superconductor_data.tsv"),
                "PIEVO_AFE_POOL_LIMIT": "50000",
                "PIEVO_AFE_POOL_PROPOSALS": "256",
            }
        )
    else:
        environment.update({"PIEVO_MBO_FORBIDDEN_CANDIDATES": "", "PIEVO_AFE_POOL_PATH": "", "PIEVO_AFE_POOL_LIMIT": "0", "PIEVO_AFE_POOL_PROPOSALS": "0"})
    if task == "SPO":
        environment.update({"PIEVO_ALGORITHM_ID": "radp_search_afe_spo_v3", "PIEVO_RADP_SEARCH_STRATEGY": "afe_spo_v2", "PIEVO_AFE_SCHEMA_VERSION": "spo_v2"})
    else:
        environment.update({"PIEVO_ALGORITHM_ID": "radp_search_afe_v9", "PIEVO_RADP_SEARCH_STRATEGY": "afe_v9", "PIEVO_AFE_SCHEMA_VERSION": "v9"})
    environment["PIEVO_RADP_WIDE_SEED_BANK"] = "0"
    environment["PIEVO_AFE_PORTFOLIO_SIZE"] = "18"
    environment["PIEVO_SOURCE_GATE"] = "1" if method == "radp" else "0"
    environment["PIEVO_SOURCE_GATE_MODE"] = "active" if method == "radp" else "off"
    environment["PIEVO_SOURCE_GATE_PROVIDER"] = f"official_{task.lower()}" if method == "radp" else "none"
    environment["PIEVO_REVISION_PROTOCOL"] = "1" if method == "radp" else "0"
    environment["PIEVO_ACTIVE_BACKTRACK"] = "1" if method == "radp" else "0"
    environment["PIEVO_ACCEPTED_PRINCIPLE_ONLY"] = "1" if method == "radp" else "0"
    environment["PIEVO_GUIDANCE_ONLY_CONTEXT"] = "1"
    environment["PIEVO_NEUTRAL_REWARD_GUIDED"] = "1" if method == "radp" and task == "MBO" else "0"
    environment["PIEVO_SOURCE_GATE_MAX_CALLS"] = "8" if method == "radp" else "0"
    environment["PIEVO_SOURCE_GATE_TOLERANCE"] = "1e-6"
    environment["PIEVO_MBO_INDEPENDENT_VERIFIER"] = ""
    # The recheck uses the released PiEvo native generator. The deterministic
    # common stream remains available only as an explicit diagnostic mode.
    use_common_generator = method == "pievo" and pievo_generator == "common"
    environment["PIEVO_COMMON_GENERATOR"] = "1" if use_common_generator else "0"
    environment["PIEVO_OUTCOME_BLIND_RECOVERY"] = "1" if recovery_contract != "none" else "0"
    environment["PIEVO_RECOVERY_CONTRACT_ID"] = recovery_contract
    environment["PIEVO_MBO_APPLICABILITY_DOMAIN"] = applicability_domain if task == "MBO" else "none"
    environment["PIEVO_GENERATOR_ID"] = (
        "radp_common_generator_v1"
        if use_common_generator
        else (
            "pievo_native_release_v1"
            if method == "pievo"
            else (
            "radp_afe_v9"
            if method == "radp" and task == "MBO"
            else "llm_hypothesis_stream"
            )
        )
    )
    return environment


def audit_summary(output: Path, task: str, expected_primary: int) -> dict[str, Any]:
    rows = load_jsonl(output / AUDIT_NAMES[task])
    primary = [
        row for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and isinstance(row.get("output"), (int, float))
    ]
    invalid = [row for row in rows if str(row.get("tool_name", "")).startswith("evaluate_") and row.get("success") is not True]
    duplicate = [row for row in rows if str(row.get("tool_name", "")).startswith("evaluate_") and row.get("duplicate_suppressed") is True]
    recovery_rows = load_jsonl(output / "outcome_blind_recovery_audit.jsonl")
    return {"audit_file": str((output / AUDIT_NAMES[task]).resolve()), "audit_rows": len(rows), "primary": len(primary), "invalid": len(invalid), "duplicate": len(duplicate), "recovery_events": len(recovery_rows), "recovery_candidates": sum(len(row.get("selected_candidates", [])) for row in recovery_rows if isinstance(row.get("selected_candidates"), list)), "expected_primary": expected_primary, "primary_complete": len(primary) == expected_primary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS), required=True)
    parser.add_argument("--method", choices=tuple(REPOS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument(
        "--pievo-generator",
        choices=("native", "common"),
        default="native",
        help="PiEvo candidate generator; common is diagnostic-only and never the default.",
    )
    parser.add_argument(
        "--allow-extended-seed",
        action="store_true",
        help="Allow a non-Table-1 seed for an isolated recheck track.",
    )
    parser.add_argument(
        "--candidate-recovery",
        choices=("none", "outcome_blind_structural_recovery_v2"),
        default="none",
        help="Enable the predeclared outcome-blind repair contract for an exhausted native proposal pool.",
    )
    parser.add_argument(
        "--run-track",
        default=TABLE1_TRACK,
        help="Immutable track label written to the run manifest.",
    )
    parser.add_argument(
        "--applicability-domain",
        choices=("none", "mbo_structural_envelope_v2"),
        default="none",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run an isolated 1-primary integration check; never eligible for Table 1/2.",
    )
    # Resolve the workstation Keychain entry by default.  A plaintext file is
    # an explicit opt-in for an isolated legacy run.
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    args = parser.parse_args()
    decode_protocol = load_decode_protocol(CONTROLLED_PROTOCOL)
    if not args.smoke and args.budget != 24:
        raise SystemExit("formal Gemini Table 1 cells require exactly 24 primary evaluations")
    if args.smoke and args.budget != 1:
        raise SystemExit("Gemini smoke cells require exactly 1 primary evaluation")
    if args.seed < 0:
        raise SystemExit("seed must be non-negative")
    if args.seed not in {0, 1, 2} and not args.allow_extended_seed:
        raise SystemExit("extended seeds require --allow-extended-seed")
    if args.method == "radp" and args.pievo_generator != "native":
        raise SystemExit("--pievo-generator applies only to PiEvo")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be fresh: {output}")
    # Keep the venv launcher path itself.  ``Path.resolve()`` follows the
    # macOS venv symlink and bypasses its site-packages directory.
    python_executable = args.python.absolute()
    if not python_executable.is_file():
        raise SystemExit("missing Python interpreter")
    output.mkdir(parents=True, exist_ok=True)
    repo = REPOS[args.method]
    audit_temp = tempfile.NamedTemporaryFile(prefix="gemini_pievo_no_thinking_", suffix=".jsonl", delete=False)
    audit_temp.close()
    bridge_command = [str(python_executable), str(BRIDGE)]
    if args.api_key_file is not None:
        bridge_command.extend(["--api-key-file", str(args.api_key_file.resolve())])
    bridge_command.extend(["--model", NATIVE_MODEL, "--port", "0", "--audit-log", audit_temp.name])
    bridge = subprocess.Popen(
        bridge_command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    process_returncode = 1
    bridge_url = ""
    task_config_copy = output / TASK_CONFIGS[args.task].name
    console_log = output / "pievo_console.log"
    started = datetime.now(timezone.utc).isoformat()
    try:
        ready = parse_ready(bridge)
        if ready.get("model") != NATIVE_MODEL:
            raise RuntimeError(
                f"Gemini bridge model mismatch: expected {NATIVE_MODEL}, got {ready.get('model')}"
            )
        bridge_url = ready["base_url"]
        write_gemini_task_config(TASK_CONFIGS[args.task], task_config_copy, bridge_url)
        environment = configure_environment(
            args.task,
            args.method,
            args.seed,
            args.budget,
            output,
            repo,
            bridge_url,
            smoke=args.smoke,
            pievo_generator=args.pievo_generator,
            recovery_contract=args.candidate_recovery,
            run_track=(SMOKE_TRACK if args.smoke else args.run_track),
            applicability_domain=args.applicability_domain,
        )
        command = [
            str(python_executable), "-m", "pievo.main",
            "--task_config", str(task_config_copy),
            "--model_config", str(MODEL_CONFIGS[args.task]),
            "--output_dir", str(output),
            "--max_turn", "280",
            "--warm_up_rounds", "5",
            "--anomaly_threshold", "0.85",
            "--anomaly_cnt_threshold", "3",
            "--budget", str(args.budget + 16),
            "--agent_api_key_force", "local-bridge",
            "--agent_base_url_force", bridge_url,
            "--agent_model_name_force", MODEL_ALIAS,
            "--agent_temperature_force", str(decode_protocol["temperature"]),
            "--agent_max_tokens_force", str(decode_protocol["max_tokens"]),
        ]
        with console_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        process_returncode = completed.returncode
    finally:
        bridge.terminate()
        try:
            bridge.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bridge.kill()
            bridge.wait(timeout=10)
    shutil.copy2(audit_temp.name, output / "gemini_no_thinking_bridge.jsonl")
    Path(audit_temp.name).unlink(missing_ok=True)
    bridge_rows = load_jsonl(output / "gemini_no_thinking_bridge.jsonl")
    bridge_model_ok = bool(bridge_rows) and all(
        row.get("native_model") == NATIVE_MODEL for row in bridge_rows
    )
    bridge_ok = bool(bridge_rows) and all(row.get("no_thinking_verified") is True for row in bridge_rows)
    bridge_seed_ok = bool(bridge_rows) and all(
        row.get("seed_forwarded") is True and row.get("seed_requested") == args.seed
        for row in bridge_rows
    )
    summary = audit_summary(output, args.task, args.budget)
    status = "completed" if (
        process_returncode == 0
        and summary["primary_complete"]
        and bridge_model_ok
        and bridge_ok
        and bridge_seed_ok
    ) else "failed"
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": status,
        "created_at_utc": started,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": 0 if status == "completed" else process_returncode or 1,
        "run_identity": {"task": args.task, "method": args.method, "seed": args.seed, "primary_budget": args.budget, "model": MODEL_ALIAS, "native_model": NATIVE_MODEL, "track": SMOKE_TRACK if args.smoke else args.run_track, "stage": "smoke" if args.smoke else "formal", "generator_id": environment.get("PIEVO_GENERATOR_ID"), "pievo_generator_mode": args.pievo_generator if args.method == "pievo" else None, "candidate_recovery_contract": args.candidate_recovery, "applicability_domain_contract": args.applicability_domain},
        "runtime": {"transport": "local OpenAI-compatible bridge to New API Gemini native /v1beta", "bridge_base_url": bridge_url, "native_model": NATIVE_MODEL, "thinking_budget": 0, "temperature": decode_protocol["temperature"], "max_tokens": decode_protocol["max_tokens"], "top_p": decode_protocol["top_p"], "top_k": decode_protocol["top_k"], "presence_penalty": decode_protocol["presence_penalty"], "repetition_penalty": decode_protocol["repetition_penalty"], "enable_thinking": decode_protocol["enable_thinking"], "context_messages": decode_protocol["context_messages"], "all_bridge_calls_native_model": bridge_model_ok, "all_bridge_calls_no_thinking": bridge_ok, "all_bridge_calls_forwarded_seed": bridge_seed_ok, "provider_seed_supported": None, "seed_transport_attested": bridge_seed_ok, "seed_provider_determinism_attested": False, "seed_requested": args.seed, "primary": summary["primary"], "invalid": summary["invalid"], "duplicate": summary["duplicate"], "recovery_events": summary["recovery_events"], "recovery_candidates": summary["recovery_candidates"]},
        "paths": {"task_config": str(task_config_copy.resolve()), "model_config": str(MODEL_CONFIGS[args.task].resolve()), "official_repo": str(repo.resolve()), "console_log": str(console_log.resolve()), "bridge_audit": str((output / "gemini_no_thinking_bridge.jsonl").resolve()), "decode_protocol": str(CONTROLLED_PROTOCOL.resolve())},
        "hashes": {"task_config_sha256": sha256(task_config_copy), "model_config_sha256": sha256(MODEL_CONFIGS[args.task]), "official_repo_main_sha256": sha256(repo / "pievo/main.py"), "bridge_sha256": sha256(BRIDGE), "decode_protocol_sha256": sha256(CONTROLLED_PROTOCOL)},
        "evidence": {"official_released_evaluator": True, "matched_qwen_hidden_paper_assets": False, "claim_boundary": "Integration smoke only; excluded from Table 1/2." if args.smoke else "Gemini no-thinking matched released-evaluator recheck; not exact hidden paper leaderboard recovery.", "candidate_recovery_outcome_blind": args.candidate_recovery != "none"},
        "audit": summary,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "task": args.task, "method": args.method, "seed": args.seed, "primary": summary["primary"], "bridge_calls": len(bridge_rows), "native_model_verified": bridge_model_ok, "no_thinking_verified": bridge_ok, "seed_forwarded": bridge_seed_ok, "process_returncode": process_returncode}, ensure_ascii=False, indent=2))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
