#!/usr/bin/env python3
"""Fail-closed runtime contract for matched PiEvo/RADP experiments.

The old launcher selected a directory named ``radp`` but never proved that
the RADP policy was enabled.  This module defines the method-specific runtime
contract and produces an attestation that can be checked before an API call
and again when metrics are reduced.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from independent_verifier_contract import load_and_validate_provider_attestation
except ImportError:  # imported as experiments.formal_runtime_contract
    from experiments.independent_verifier_contract import (
        load_and_validate_provider_attestation,
    )


SCHEMA_VERSION = "matched_runtime_attestation.v2"
APPROVAL_SCHEMA_VERSION = "matched_formal_approval.v2"
TASKS = ("NHO", "MBO", "SPO", "TMC")
METHODS = ("pievo", "radp")
EMBEDDING_REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json")
DEFAULT_EMBEDDING_MODEL_PATH = Path(
    "/Users/michael/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/"
    "snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
)
RADP_REQUIRED_FEATURES = (
    "gate_modes",
    "search_verification_memory_separation",
    "source_posterior",
    "cost_aware_action_selection",
    "provisional_principles",
    "backtrack",
    "verification_cost_accounting",
    "main_loop_integration_audit",
    "afe_acquisition",
    "outcome_blind_initial_priors",
)
CODE_PATHS = {
    "main": "pievo/main.py",
    "evolveflow": "pievo/group/evolveflow.py",
    "manage": "pievo/group/manage.py",
    "agent_abstract": "pievo/agents/abstract.py",
    "experiment_agent": "pievo/agents/experiment.py",
    "instruction": "pievo/word/instruction.py",
    "embedding": "pievo/word/embedding.py",
    "model_config": "config/model.yaml",
    "chembl_tool": "pievo/tools/tool_chembl_official.py",
    "scientific_tool": "pievo/tools/tool_scientific_official.py",
    "candidate_validation": "pievo/tools/candidate_validation.py",
    "radp_core": "pievo/radp/core.py",
    "radp_integration": "pievo/radp/integration.py",
    "radp_afe": "pievo/radp/afe.py",
    "radp_wide_candidate": "pievo/radp/wide_candidate.py",
    "radp_capabilities": "pievo/radp/capabilities.json",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def env_bool(environment: Mapping[str, str], name: str) -> bool:
    return str(environment.get(name, "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_int(environment: Mapping[str, str], name: str, default: int = 0) -> int:
    try:
        return int(environment.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def embedding_model_from_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    """Return an offline, content-addressed attestation for the local encoder."""
    raw_path = str(
        environment.get("PIEVO_EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL_PATH)
    ).strip()
    model_path = Path(raw_path) if raw_path else None
    file_hashes = {
        name: sha256(model_path / name) if model_path is not None else None
        for name in EMBEDDING_REQUIRED_FILES
    }
    offline = str(environment.get("PIEVO_EMBEDDING_OFFLINE", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    complete = bool(
        model_path is not None
        and model_path.is_dir()
        and all(isinstance(value, str) and len(value) == 64 for value in file_hashes.values())
    )
    return {
        "path": str(model_path.resolve()) if model_path is not None else "",
        "offline": offline,
        "file_hashes": file_hashes,
        "passed": complete and offline,
    }


def expected_runtime(method: str, task: str, seed: int) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if method == "pievo":
        return {
            "algorithm_id": "pievo_matched_release_v1",
            "source_gate_enabled": False,
            "gate_mode": "off",
            "provider": "none",
            "revision_protocol_enabled": False,
            "backtrack_enabled": False,
            "guidance_only_context": False,
            "llm_seed": seed,
        }
    return {
        "algorithm_id": (
            "radp_search_afe_spo_v3" if task == "SPO" else "radp_search_afe_v9"
        ),
        "source_gate_enabled": True,
        "gate_mode": "active",
        "provider": f"official_{task.lower()}",
        "revision_protocol_enabled": True,
        "backtrack_enabled": True,
        "guidance_only_context": True,
        "llm_seed": seed,
    }


def runtime_from_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    try:
        llm_seed = int(environment.get("PIEVO_SEED", "-1"))
    except (TypeError, ValueError):
        llm_seed = -1
    return {
        "algorithm_id": str(environment.get("PIEVO_ALGORITHM_ID", "")),
        "source_gate_enabled": env_bool(environment, "PIEVO_SOURCE_GATE"),
        "gate_mode": str(environment.get("PIEVO_SOURCE_GATE_MODE", "off")).strip().lower(),
        "provider": str(environment.get("PIEVO_SOURCE_GATE_PROVIDER", "none")).strip().lower(),
        "revision_protocol_enabled": env_bool(environment, "PIEVO_REVISION_PROTOCOL"),
        "backtrack_enabled": env_bool(environment, "PIEVO_ACTIVE_BACKTRACK"),
        "guidance_only_context": env_bool(
            environment, "PIEVO_GUIDANCE_ONLY_CONTEXT"
        ),
        "independent_verifier": str(
            environment.get("PIEVO_MBO_INDEPENDENT_VERIFIER", "")
        ).strip().lower(),
        "candidate_prefilter_enabled": env_bool(
            environment, "PIEVO_PREFILTER_INVALID_CANDIDATES"
        ),
        "candidate_prefilter_rejection_limit": env_int(
            environment, "PIEVO_MAX_PREFILTER_REJECTIONS"
        ),
        "candidate_prefilter_empty_pool_limit": env_int(
            environment, "PIEVO_MAX_PREFILTER_EMPTY_POOLS"
        ),
        "llm_seed": llm_seed,
    }


def repository_hashes(repository: Path, method: str) -> dict[str, str | None]:
    names = (
        CODE_PATHS
        if method == "radp"
        else {key: value for key, value in CODE_PATHS.items() if not key.startswith("radp_")}
    )
    return {name: sha256(repository / relative) for name, relative in names.items()}


def validate_runtime(
    *,
    method: str,
    task: str,
    seed: int,
    repository: Path,
    environment: Mapping[str, str] | None = None,
    input_paths: Mapping[str, Path] | None = None,
    require_formal_ready: bool = True,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    expected = expected_runtime(method, task, seed)
    if require_formal_ready and task in {"MBO", "SPO", "TMC"}:
        expected.update(
            {
                "candidate_prefilter_enabled": True,
                "candidate_prefilter_rejection_limit": 12,
                "candidate_prefilter_empty_pool_limit": 2,
            }
        )
    actual = runtime_from_environment(environment)
    # Development-only factor audit: keep the RADP AFE implementation fixed
    # while allowing the source-gate factor to be switched off or shadowed.
    # These overrides are intentionally unavailable to formal readiness checks.
    if (
        method == "radp"
        and not require_formal_ready
        and str(environment.get("PIEVO_EXPLORATORY_AFE_GATE_MODES", "0")).strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        expected.update(
            {
                "source_gate_enabled": actual.get("source_gate_enabled"),
                "gate_mode": actual.get("gate_mode"),
                "provider": actual.get("provider"),
                "revision_protocol_enabled": actual.get("revision_protocol_enabled"),
                "backtrack_enabled": actual.get("backtrack_enabled"),
                "accepted_principle_only": actual.get("accepted_principle_only"),
                "guidance_only_context": actual.get("guidance_only_context"),
                "neutral_reward_guided": actual.get("neutral_reward_guided"),
                "source_gate_max_calls": actual.get("source_gate_max_calls"),
                "independent_verifier": actual.get("independent_verifier"),
            }
        )
    errors: list[str] = []
    embedding_model = embedding_model_from_environment(environment)
    if not embedding_model["passed"]:
        errors.append(
            "local Qwen embedding model must be complete, content-addressed, and offline"
        )
    strategy = str(
        environment.get("PIEVO_RADP_SEARCH_STRATEGY", "legacy_v1")
    ).strip().lower()
    expected_strategy = "afe_spo_v2" if task == "SPO" else "afe_v9"
    if method == "radp" and require_formal_ready and strategy != expected_strategy:
        errors.append(
            f"formal RADP runs must use the frozen {expected_strategy} acquisition strategy"
        )
    for key, value in expected.items():
        if actual.get(key) != value:
            errors.append(
                f"runtime mismatch for {key}: expected {value!r}, got {actual.get(key)!r}"
            )
    hashes = repository_hashes(repository, method)
    for name, path in (input_paths or {}).items():
        hashes[f"input_{name}"] = sha256(path)
    missing_code = sorted(name for name, digest in hashes.items() if digest is None)
    if missing_code:
        errors.append(f"required code paths are missing: {missing_code}")

    capabilities_path = repository / CODE_PATHS["radp_capabilities"]
    capabilities = load_json(capabilities_path, {}) if method == "radp" else {}
    task_capabilities: dict[str, Any] = {}
    provider_attestation: dict[str, Any] | None = None
    if method == "radp":
        if not isinstance(capabilities, dict):
            capabilities = {}
        capability_ids = capabilities.get("algorithm_ids")
        if not isinstance(capability_ids, list):
            capability_ids = [capabilities.get("algorithm_id")]
        if expected["algorithm_id"] not in capability_ids:
            errors.append("RADP capability manifest has the wrong algorithm_id")
        features = capabilities.get("features", {})
        missing_features = [
            feature
            for feature in RADP_REQUIRED_FEATURES
            if not isinstance(features, dict) or features.get(feature) is not True
        ]
        if missing_features:
            errors.append(f"RADP required features are not implemented: {missing_features}")
        tasks = capabilities.get("tasks", {})
        task_capabilities = tasks.get(task, {}) if isinstance(tasks, dict) else {}
        if not isinstance(task_capabilities, dict):
            task_capabilities = {}
        if task_capabilities.get("provider") != expected["provider"]:
            errors.append("RADP task provider does not match the runtime contract")
        # ``formal_ready`` remains reserved for an independent verifier claim.
        # The matched performance route has a separate readiness flag and may
        # proceed without silently relabeling the released surrogate.
        formal_integration = capabilities.get("formal_integration", {})
        if require_formal_ready and (
            not isinstance(formal_integration, dict)
            or formal_integration.get("source_gate_main_loop") is not True
            or formal_integration.get("search_history_replacement") is not True
            or formal_integration.get("principle_lifecycle_main_loop") is not True
            or formal_integration.get("accepted_principle_gp_backtrack") is not True
        ):
            errors.append("RADP principle lifecycle/GP backtrack integration is not formal-ready")
        if require_formal_ready and task_capabilities.get("matched_evaluator_ready") is not True:
            errors.append(f"matched released evaluator for {task} is not ready")

    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "task": task,
        "seed": seed,
        "repository": str(repository.resolve()),
        "expected_runtime": expected,
        "actual_runtime": actual,
        "embedding_model": embedding_model,
        "code_hashes": hashes,
        "capabilities_path": str(capabilities_path.resolve()) if method == "radp" else None,
        "capabilities_sha256": sha256(capabilities_path) if method == "radp" else None,
        "task_capabilities": task_capabilities,
        "provider_attestation": provider_attestation,
        "require_formal_ready": require_formal_ready,
        "errors": errors,
        "passed": not errors,
    }


def attestation_is_formal(value: Any, *, method: str, task: str, seed: int) -> bool:
    if not isinstance(value, dict):
        return False
    code_hashes = value.get("code_hashes", {})
    expected = value.get("expected_runtime")
    actual = value.get("actual_runtime")
    embedding_model = value.get("embedding_model")
    # The launcher records these prefilter settings for every task.  They are
    # required for formal tasks with a deterministic candidate validator and
    # remain optional metadata for NHO.
    optional_runtime_fields = {
        "independent_verifier",
        "candidate_prefilter_enabled",
        "candidate_prefilter_rejection_limit",
        "candidate_prefilter_empty_pool_limit",
    }
    hashes_valid = bool(
        isinstance(code_hashes, dict)
        and code_hashes
        and all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in code_hashes.values()
        )
    )
    runtime_valid = bool(
        isinstance(expected, dict)
        and isinstance(actual, dict)
        and all(actual.get(key) == expected_value for key, expected_value in expected.items())
        and set(actual).issubset(set(expected) | optional_runtime_fields)
    )
    embedding_valid = bool(
        isinstance(embedding_model, dict)
        and embedding_model.get("passed") is True
        and embedding_model.get("offline") is True
        and isinstance(embedding_model.get("path"), str)
        and embedding_model.get("path")
        and isinstance(embedding_model.get("file_hashes"), dict)
        and set(EMBEDDING_REQUIRED_FILES).issubset(embedding_model["file_hashes"])
        and all(
            isinstance(embedding_model["file_hashes"].get(name), str)
            and len(embedding_model["file_hashes"][name]) == 64
            and all(character in "0123456789abcdef" for character in embedding_model["file_hashes"][name])
            for name in EMBEDDING_REQUIRED_FILES
        )
    )
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("method") == method
        and value.get("task") == task
        and value.get("seed") == seed
        and value.get("require_formal_ready") is True
        and value.get("passed") is True
        and not value.get("errors")
        and runtime_valid
        and embedding_valid
        and hashes_valid
    )


def formal_approval_is_valid(
    value: Any,
    *,
    method: str,
    task: str,
    code_hashes: Mapping[str, str | None],
    embedding_model: Mapping[str, Any],
) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        value.get("schema_version") == APPROVAL_SCHEMA_VERSION
        and value.get("method") == method
        and value.get("task") == task
        and value.get("passed") is True
        and not value.get("errors")
        and value.get("code_hashes") == dict(code_hashes)
        and value.get("embedding_model") == dict(embedding_model)
        and isinstance(value.get("development_manifest_sha256"), str)
        and len(value["development_manifest_sha256"]) == 64
    )
