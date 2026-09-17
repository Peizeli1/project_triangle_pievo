#!/usr/bin/env python3
"""Freeze a non-Qwen model track after bounded development evidence.

This is an offline provenance artifact.  It binds the model track to the
decode protocol, launcher, runtime code, and the development approval before
any formal API collection starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-track", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--development-approval", required=True, type=Path)
    parser.add_argument("--runtime-attestation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    approval = load(args.development_approval)
    attestation = load(args.runtime_attestation)
    protocol_hash = sha256(args.protocol)
    runner_hash = sha256(args.runner)
    approval_hash = sha256(args.development_approval)
    attestation_hash = sha256(args.runtime_attestation)
    code_hashes = attestation.get("code_hashes", {})
    embedding_model = attestation.get("embedding_model", {})
    errors: list[str] = []
    if approval.get("passed") is not True:
        errors.append("development approval did not pass")
    if approval.get("model") != args.model:
        errors.append("development approval model mismatch")
    if approval.get("model_track") != args.model_track:
        errors.append("development approval model-track mismatch")
    if approval.get("protocol_sha256") != protocol_hash:
        errors.append("development approval protocol hash mismatch")
    if attestation.get("passed") is not True:
        errors.append("runtime attestation did not pass")
    if not isinstance(code_hashes, dict) or not code_hashes:
        errors.append("runtime code hashes are missing")
    if not isinstance(embedding_model, dict) or embedding_model.get("passed") is not True:
        errors.append("embedding-model attestation did not pass")
    result = {
        "schema_version": "model_track_protocol_freeze.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_track": args.model_track,
        "protocol_sha256": protocol_hash,
        "runner_sha256": runner_hash,
        "development_approval_sha256": approval_hash,
        "development_attestation_sha256": attestation_hash,
        "development_approval": str(args.development_approval.resolve()),
        "development_attestation": str(args.runtime_attestation.resolve()),
        "code_hashes": code_hashes,
        "embedding_model": embedding_model,
        "errors": errors,
        "passed": not errors,
        "claim_boundary": "Frozen model-track provenance; no performance or SOTA claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 5


if __name__ == "__main__":
    raise SystemExit(main())
