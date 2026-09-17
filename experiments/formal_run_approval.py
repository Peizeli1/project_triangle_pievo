#!/usr/bin/env python3
"""Create a code-bound formal approval from one bounded development run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from formal_runtime_contract import APPROVAL_SCHEMA_VERSION
except ImportError:
    from experiments.formal_runtime_contract import APPROVAL_SCHEMA_VERSION


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def radp_audit_ok(path: Path) -> bool:
    events: set[str] = set()
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get("schema_version") == "radp_runtime_audit.v1":
            events.add(str(value.get("event")))
    # A clean AFE development run intentionally keeps unverified principles in
    # PROPOSED.  Requiring a principle transition here would force a false
    # lifecycle event and would reject the accepted-principle-only contract.
    return {"runtime_start", "gate_decision", "gate_finalized"} <= events


def radp_afe_audit_ok(path: Path) -> bool:
    accepted_schema_pairs = (
        {"radp_afe_portfolio.v3", "radp_afe_acquisition.v3"},
        {"radp_afe_portfolio.v8", "radp_afe_acquisition.v8"},
        {"radp_afe_portfolio.spo_v1", "radp_afe_acquisition.spo_v1"},
        {"radp_afe_portfolio.v9", "radp_afe_acquisition.v9"},
        {"radp_afe_portfolio.spo_v2", "radp_afe_acquisition.spo_v2"},
    )
    schemas: set[str] = set()
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        schema = value.get("schema_version")
        if any(schema in pair for pair in accepted_schema_pairs):
            schemas.add(str(schema))
    return any(pair <= schemas for pair in accepted_schema_pairs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.development_run / "run_manifest.json"
    manifest = load_json(manifest_path)
    identity = manifest.get("run_identity", {})
    runtime = manifest.get("runtime", {})
    evidence = manifest.get("final_evidence", {})
    method = identity.get("method")
    task = identity.get("task")
    model = identity.get("model")
    model_track = identity.get("model_track")
    budget = identity.get("primary_budget")
    errors: list[str] = []
    if manifest.get("schema_version") != "matched_qwen235b_run_manifest.v3":
        errors.append("development manifest is not schema v3")
    protocol_sha256 = manifest.get("hashes", {}).get("protocol_sha256")
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        errors.append("development protocol hash is missing")
    if manifest.get("status") != "completed" or identity.get("stage") != "development":
        errors.append("development run is not completed development evidence")
    if not isinstance(budget, int) or not 6 <= budget <= 8:
        errors.append("development run must use 6 to 8 primary evaluations")
    if runtime.get("runtime_attestation_passed") is not True:
        errors.append("development runtime attestation did not pass")
    if runtime.get("runtime_attestation_formal") is not False:
        errors.append("development attestation has the wrong readiness scope")
    embedding_model = runtime.get("embedding_model", {})
    embedding_hashes = embedding_model.get("file_hashes", {}) if isinstance(embedding_model, dict) else {}
    if not (
        isinstance(embedding_model, dict)
        and embedding_model.get("passed") is True
        and embedding_model.get("offline") is True
        and isinstance(embedding_model.get("path"), str)
        and embedding_model.get("path")
        and isinstance(embedding_hashes, dict)
        and all(isinstance(embedding_hashes.get(name), str) and len(embedding_hashes[name]) == 64 for name in ("config.json", "model.safetensors", "tokenizer.json"))
    ):
        errors.append("development embedding-model attestation is missing or incomplete")
    if evidence.get("seed_application_passed") is not True:
        errors.append("seed application proof is missing or invalid")
    if evidence.get("token_accounting_present") is not True:
        errors.append("token accounting is missing")
    if evidence.get("token_ceiling_passed") is not True:
        errors.append("development token ceiling was exceeded")
    if model == "gpt-5.6-luna":
        if runtime.get("outcome_blind_recovery_enabled") is not True:
            errors.append("Luna development run lacks the registered recovery contract")
        if runtime.get("outcome_blind_recovery_contract") != "outcome_blind_structural_recovery_v2":
            errors.append("Luna development run has the wrong recovery contract")
    audit = evidence.get("audit", {})
    if audit.get("primary") != budget:
        errors.append("development evaluator count does not match its budget")
    if audit.get("invalid", 0) > runtime.get("invalid_attempt_limit", 0):
        errors.append("development invalid-attempt limit was exceeded")
    if method == "radp":
        if not radp_audit_ok(args.development_run / "radp_runtime_audit.jsonl"):
            errors.append(
                "RADP development run lacks startup, gate-decision, or finalized-gate evidence"
            )
        if not radp_afe_audit_ok(args.development_run / "radp_candidate_portfolio_audit.jsonl"):
            errors.append("RADP development run lacks AFE portfolio/acquisition evidence")
    code_hashes = manifest.get("hashes", {}).get("attested_code_hashes", {})
    if not isinstance(code_hashes, dict) or not code_hashes or any(not value for value in code_hashes.values()):
        errors.append("development code hashes are incomplete")

    approval = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "method": method,
        "task": task,
        "model": model,
        "model_track": model_track,
        "protocol_sha256": manifest.get("hashes", {}).get("protocol_sha256"),
        "development_seed": identity.get("seed"),
        "development_budget": budget,
        "development_manifest": str(manifest_path.resolve()),
        "development_manifest_sha256": sha256(manifest_path),
        "code_hashes": code_hashes,
        "embedding_model": embedding_model,
        "errors": errors,
        "passed": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 5


if __name__ == "__main__":
    raise SystemExit(main())
