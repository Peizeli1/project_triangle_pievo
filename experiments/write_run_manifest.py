#!/usr/bin/env python3
"""Write or update one auditable matched-run manifest.

The manifest contains no API key.  It is intentionally usable before a run
starts and after the runner exits, including when the underlying run fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from experiments.formal_usage import read_cumulative_usage
except ImportError:
    from formal_usage import read_cumulative_usage


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def audit_summary(output: Path, task: str) -> dict[str, int | bool]:
    names = {
        "NHO": "official_nho_tool_audit.jsonl",
        "MBO": "official_chembl_tool_audit.jsonl",
        "SPO": "official_spo_tool_audit.jsonl",
        "TMC": "official_tmc_tool_audit.jsonl",
    }
    path = output / names[task]
    primary = invalid = duplicate = verification = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            tool = str(row.get("tool_name", ""))
            if tool.startswith("verify_") or row.get("verification") is True:
                verification += 1
            elif tool.startswith("evaluate_") and row.get("duplicate_suppressed") is True:
                duplicate += 1
            elif tool.startswith("evaluate_") and row.get("success") is True:
                primary += 1
            elif tool.startswith("evaluate_"):
                invalid += 1
    prefilter_path = output / "candidate_prefilter_audit.jsonl"
    prefilter_invalid = prefilter_duplicate = 0
    if prefilter_path.is_file():
        for line in prefilter_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("disposition") == "invalid":
                prefilter_invalid += 1
            elif row.get("disposition") == "canonical_duplicate":
                prefilter_duplicate += 1
    afe_proposal_path = output / "radp_afe_proposal_audit.jsonl"
    afe_controller_invalid = afe_controller_duplicate = 0
    if afe_proposal_path.is_file():
        for line in afe_proposal_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("disposition") == "afe_invalid_proposal":
                afe_controller_invalid += 1
            elif row.get("disposition") == "afe_canonical_duplicate":
                afe_controller_duplicate += 1
    return {
        "audit_present": path.is_file(),
        "primary": primary,
        "invalid": invalid,
        "duplicate": duplicate,
        "verification": verification,
        "prefilter_audit_present": prefilter_path.is_file(),
        "prefilter_invalid": prefilter_invalid,
        "prefilter_duplicate": prefilter_duplicate,
        "afe_controller_audit_present": afe_proposal_path.is_file(),
        "afe_controller_invalid": afe_controller_invalid,
        "afe_controller_duplicate": afe_controller_duplicate,
        "effective_invalid": invalid + prefilter_invalid,
        "effective_duplicate": duplicate + prefilter_duplicate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=["NHO", "MBO", "SPO", "TMC"])
    parser.add_argument("--method", required=True, choices=["pievo", "radp"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--formal-approval", type=Path, default=None)
    parser.add_argument("--stage", choices=["development", "formal"], required=True)
    parser.add_argument("--token-ceiling", type=int, required=True)
    parser.add_argument("--invalid-limit", type=int, required=True)
    parser.add_argument("--status", choices=["started", "completed", "failed"], required=True)
    parser.add_argument("--exit-code", type=int, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}

    now = datetime.now(timezone.utc).isoformat()
    attestation = load_json(args.runtime_attestation)
    embedding_model = attestation.get("embedding_model", {})
    if not isinstance(embedding_model, dict):
        embedding_model = {}
    embedding_hashes = embedding_model.get("file_hashes", {})
    if not isinstance(embedding_hashes, dict):
        embedding_hashes = {}
    protocol_data = load_json(args.protocol)
    decode_protocol_path = Path(os.getenv("PIEVO_DECODE_PROTOCOL", ""))
    decode_protocol = load_json(decode_protocol_path) if str(decode_protocol_path) else {}
    prefilter_policy = (
        protocol_data.get("minimal_controlled_api_screen", {})
        .get("candidate_prefilter", {})
    )
    seed_path = args.output / "seed_application.json"
    seed_proof = load_json(seed_path)
    token_usage = read_cumulative_usage(args.output / "token_usage.json")
    runtime_audit_path = args.output / "radp_runtime_audit.jsonl"
    portfolio_audit_path = args.output / "radp_candidate_portfolio_audit.jsonl"
    common_generator_audit_path = args.output / "common_candidate_generator_audit.jsonl"
    context_boundary_audit_path = args.output / "agent_context_boundary_audit.jsonl"
    prefilter_audit_path = args.output / "candidate_prefilter_audit.jsonl"
    afe_proposal_audit_path = args.output / "radp_afe_proposal_audit.jsonl"
    chembl37_audit_path = args.output / "chembl37_verifier_audit.jsonl"
    approval_path = args.formal_approval
    audit = audit_summary(args.output, args.task)
    manifest: dict[str, Any] = {
        "schema_version": "matched_qwen235b_run_manifest.v3",
        "status": args.status,
        "created_at_utc": existing.get("created_at_utc", now),
        "updated_at_utc": now,
        "exit_code": args.exit_code,
        "run_identity": {
            "task": args.task,
            "method": args.method,
            "seed": args.seed,
            "primary_budget": args.budget,
            "model": os.getenv("PIEVO_AGENT_MODEL", "qwen3-235b-a22b-instruct-2507"),
            "model_track": os.getenv("PIEVO_MODEL_TRACK", ""),
            "track": os.getenv("PIEVO_RUN_TRACK", "matched_qwen235b"),
            "stage": args.stage,
        },
        "paths": {
            "output_dir": str(args.output.resolve()),
            "official_repo": str(args.official_repo.resolve()),
            "task_config": str(args.task_config.resolve()),
            "model_config": str(args.model_config.resolve()),
            "protocol": str(args.protocol.resolve()),
            "decode_protocol": str(decode_protocol_path.resolve()) if decode_protocol_path.is_file() else None,
            "runtime_attestation": str(args.runtime_attestation.resolve()),
            "radp_candidate_portfolio_audit": str(portfolio_audit_path.resolve()),
            "common_candidate_generator_audit": str(common_generator_audit_path.resolve()),
            "candidate_prefilter_audit": str(prefilter_audit_path.resolve()),
            "radp_afe_proposal_audit": str(afe_proposal_audit_path.resolve()),
            "chembl37_verifier_audit": str(chembl37_audit_path.resolve()),
            "agent_context_boundary_audit": str(
                context_boundary_audit_path.resolve()
            ),
            "formal_approval": str(approval_path.resolve()) if approval_path else None,
            "surrogate_root": os.getenv("PIEVO_SURROGATE_ROOT"),
            "chembl_model": os.getenv("PIEVO_CHEMBL_MODEL_PATH"),
            "embedding_model": embedding_model.get("path", ""),
            "afe_pool": os.getenv("PIEVO_AFE_POOL_PATH", ""),
        },
        "hashes": {
            "task_config_sha256": sha256(args.task_config),
            "model_config_sha256": sha256(args.model_config),
            "protocol_sha256": sha256(args.protocol),
            "decode_protocol_sha256": sha256(decode_protocol_path) if decode_protocol_path.is_file() else None,
            "chembl_model_sha256": sha256(Path(os.getenv("PIEVO_CHEMBL_MODEL_PATH", ""))),
            "embedding_model_config_sha256": embedding_hashes.get("config.json"),
            "embedding_model_weights_sha256": embedding_hashes.get("model.safetensors"),
            "embedding_model_tokenizer_sha256": embedding_hashes.get("tokenizer.json"),
            "official_repo_git_head": git_head(args.official_repo),
            "runtime_attestation_sha256": sha256(args.runtime_attestation),
            "formal_approval_sha256": sha256(approval_path) if approval_path else None,
            "seed_application_sha256": sha256(seed_path),
            "radp_runtime_audit_sha256": sha256(runtime_audit_path),
            "radp_candidate_portfolio_audit_sha256": sha256(portfolio_audit_path),
            "common_candidate_generator_audit_sha256": sha256(common_generator_audit_path),
            "candidate_prefilter_audit_sha256": sha256(prefilter_audit_path),
            "radp_afe_proposal_audit_sha256": sha256(afe_proposal_audit_path),
            "chembl37_verifier_audit_sha256": sha256(chembl37_audit_path),
            "agent_context_boundary_audit_sha256": sha256(
                context_boundary_audit_path
            ),
            "attested_code_hashes": attestation.get("code_hashes", {}),
        },
        "runtime": {
            "python": os.getenv("PIEVO_PYTHON"),
            "base_url": os.getenv("PIEVO_AGENT_BASE_URL", "https://zjuapi.com/v1"),
            "max_turn": int(os.getenv("PIEVO_MAX_TURN", "280")),
            "warm_up_rounds": int(os.getenv("PIEVO_WARMUP_ROUNDS", "5")),
            "group_budget": int(os.getenv("PIEVO_GROUP_BUDGET", str(args.budget + 16))),
            "algorithm_id": os.getenv("PIEVO_ALGORITHM_ID", ""),
            "common_generator_enabled": os.getenv("PIEVO_COMMON_GENERATOR", "0") == "1",
            "outcome_blind_recovery_enabled": os.getenv(
                "PIEVO_OUTCOME_BLIND_RECOVERY", "0"
            )
            == "1",
            "outcome_blind_recovery_contract": os.getenv(
                "PIEVO_RECOVERY_CONTRACT_ID", ""
            ),
            "common_generator_id": (
                "radp_common_generator_v1"
                if os.getenv("PIEVO_COMMON_GENERATOR", "0") == "1"
                else None
            ),
            "radp_search_strategy": os.getenv(
                "PIEVO_RADP_SEARCH_STRATEGY", "legacy_v1"
            ),
            "source_gate_enabled": os.getenv("PIEVO_SOURCE_GATE", "0") == "1",
            "gate_mode": os.getenv("PIEVO_SOURCE_GATE_MODE", "off"),
            "provider": os.getenv("PIEVO_SOURCE_GATE_PROVIDER", "none"),
            "independent_verifier": os.getenv("PIEVO_MBO_INDEPENDENT_VERIFIER", ""),
            "verification_cost_accounted": True,
            "revision_protocol_enabled": os.getenv("PIEVO_REVISION_PROTOCOL", "0") == "1",
            "backtrack_enabled": os.getenv("PIEVO_ACTIVE_BACKTRACK", "0") == "1",
            "accepted_principle_only_for_hypothesis_guidance": os.getenv(
                "PIEVO_ACCEPTED_PRINCIPLE_ONLY", "0"
            )
            == "1",
            "neutral_reward_guided": os.getenv(
                "PIEVO_NEUTRAL_REWARD_GUIDED", "0"
            )
            == "1",
            "guidance_only_context": os.getenv(
                "PIEVO_GUIDANCE_ONLY_CONTEXT", "0"
            )
            == "1",
            "llm_seed": args.seed,
            "generation_config": {
                "model": os.getenv("PIEVO_AGENT_MODEL", "qwen3-235b-a22b-instruct-2507"),
                "temperature": float(os.getenv("PIEVO_AGENT_TEMPERATURE", str(decode_protocol.get("temperature", 0.6)))),
                "max_tokens": int(os.getenv("PIEVO_AGENT_MAX_TOKENS", str(decode_protocol.get("max_tokens", 4096)))),
                "top_p": decode_protocol.get("top_p", 0.8),
                "top_k": decode_protocol.get("top_k", 20),
                "presence_penalty": decode_protocol.get("presence_penalty", 1.5),
                "repetition_penalty": decode_protocol.get("repetition_penalty", 1.0),
                "enable_thinking": decode_protocol.get("enable_thinking", False),
                "reasoning_effort": decode_protocol.get("reasoning_effort"),
                "context_messages": decode_protocol.get("context_messages", 10),
            },
            "utility_generation_config": {
                "model": os.getenv(
                    "PIEVO_UTIL_MODEL_FORCE",
                    os.getenv("UTIL_LLM_CONFIG_NAME", "qwen3-235b-a22b-instruct-2507"),
                ),
                "model_track": os.getenv("PIEVO_MODEL_TRACK", ""),
                "temperature": float(os.getenv("UTIL_LLM_CONFIG_TEMPERATURE", str(decode_protocol.get("temperature", 0.6)))),
                "max_tokens": int(os.getenv("UTIL_LLM_CONFIG_MAX_TOKENS", str(decode_protocol.get("max_tokens", 4096)))),
                "top_p": decode_protocol.get("top_p", 0.8),
                "top_k": decode_protocol.get("top_k", 20),
                "presence_penalty": decode_protocol.get("presence_penalty", 1.5),
                "repetition_penalty": decode_protocol.get("repetition_penalty", 1.0),
                "enable_thinking": decode_protocol.get("enable_thinking", False),
                "reasoning_effort": decode_protocol.get("reasoning_effort"),
            },
            "decode_protocol_loaded": decode_protocol_path.is_file(),
            "token_ceiling": args.token_ceiling,
            "invalid_attempt_limit": args.invalid_limit,
            "candidate_prefilter_enabled": os.getenv(
                "PIEVO_PREFILTER_INVALID_CANDIDATES", "0"
            )
            == "1",
            "candidate_prefilter_rejection_limit": int(
                os.getenv("PIEVO_MAX_PREFILTER_REJECTIONS", "0") or 0
            ),
            "candidate_prefilter_empty_pool_limit": int(
                os.getenv("PIEVO_MAX_PREFILTER_EMPTY_POOLS", "0") or 0
            ),
            "candidate_syntax_feedback_enabled": bool(
                prefilter_policy.get("syntax_feedback_enabled", False)
            ),
            "candidate_syntax_feedback_limit": int(
                prefilter_policy.get("syntax_feedback_limit", 0) or 0
            ),
            "candidate_validated_scaffold_count": int(
                prefilter_policy.get("validated_scaffold_count", 0) or 0
            ),
            "mbo_forbidden_candidate_count": len(
                [
                    value
                    for value in os.getenv(
                        "PIEVO_MBO_FORBIDDEN_CANDIDATES", ""
                    ).split("|")
                    if value.strip()
                ]
            ),
            "runtime_attestation_passed": attestation.get("passed") is True,
            "runtime_attestation_formal": attestation.get("require_formal_ready") is True,
            "embedding_model": embedding_model,
        },
        "final_evidence": {
            "seed_application": seed_proof,
            "seed_application_passed": seed_proof.get("seed_applied") is True,
            "token_usage": token_usage,
            "token_accounting_present": token_usage["snapshot_index"] is not None,
            "token_ceiling_passed": int(token_usage["total"] or 0) <= args.token_ceiling,
            "radp_runtime_audit_present": runtime_audit_path.is_file(),
            "radp_candidate_portfolio_audit_present": portfolio_audit_path.is_file(),
            "common_candidate_generator_audit_present": common_generator_audit_path.is_file(),
            "radp_afe_proposal_audit_present": afe_proposal_audit_path.is_file(),
            "chembl37_verifier_audit_present": chembl37_audit_path.is_file(),
            "audit": audit,
        },
        "claim_boundary": {
            "official_task_assets_verified": os.getenv("PIEVO_OFFICIAL_ASSETS_VERIFIED", "0") == "1",
            "matched_evaluator_assets_verified": os.getenv("PIEVO_MATCHED_EVALUATOR_ASSETS_VERIFIED", "0") == "1",
            "exact_paper_setting_verified": False,
            "formal_comparison_scope": "matched_released_evaluator_not_exact_paper_leaderboard",
            "official_superiority_gate_passed": False,
            "apd_exact_evaluator_present": os.getenv("PIEVO_APD_EXACT_EVALUATOR_PRESENT", "0") == "1",
        },
    }
    # Preserve an immutable start timestamp and any future runner metadata.
    for key in ("started_at_utc",):
        if key in existing:
            manifest[key] = existing[key]
    if args.status == "started":
        manifest["started_at_utc"] = now
    elif existing.get("started_at_utc"):
        manifest["started_at_utc"] = existing["started_at_utc"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
