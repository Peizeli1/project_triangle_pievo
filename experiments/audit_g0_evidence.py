#!/usr/bin/env python3
"""Zero-API G0 evidence and provenance audit for the single active version."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from experiments.formal_runtime_contract import repository_hashes
except ModuleNotFoundError:
    from formal_runtime_contract import repository_hashes


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
# Retired artifacts may be recoverable through the user's OS Trash, but the
# Trash location is not a scientific input and may be hidden by macOS privacy.
# G0 therefore proves the property that matters for reproducibility: retired
# versions are absent from the active tree.  Their recorded destination is
# retained as provenance below without making filesystem visibility a gate.
TRASH = Path("/Users/michael/.Trash/ai4ai-cleanup-20260806-v7")
POOL = RESULTS / "official_chembl35_pool_seed0/chembl35_pool_seed0.jsonl"
POOL_SUMMARY = RESULTS / "official_chembl35_pool_seed0/chembl35_pool_seed0_summary.json"
PROTOCOL = ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def audit_archived_screen(path: Path) -> dict[str, Any]:
    audit_path = path / "official_mbo_development_screen_audit.json"
    audit = load(audit_path)
    methods = audit.get("methods", {})
    method_rows: dict[str, Any] = {}
    all_complete = audit.get("passed") is True
    any_stale = False
    no_partial_retry = True
    for method in ("pievo", "radp"):
        entry = methods.get(method, {})
        directory = Path(str(entry.get("directory", "")))
        # The archived audit retains the original path; the copied directory is
        # the only directory eligible for later inspection.
        local = path / method
        attestation = load(local / "runtime_attestation.json")
        manifest = load(local / "run_manifest.json")
        current_hashes = repository_hashes(
            ROOT / "official_sources" / (
                "PiEvo_radp_qwen235b" if method == "radp" else "PiEvo_matched_qwen235b"
            ),
            method,
        )
        attested = attestation.get("code_hashes", {})
        code_fresh = bool(
            isinstance(attested, dict)
            and all(attested.get(name) == digest for name, digest in current_hashes.items())
        )
        for candidate in local.rglob("*"):
            if candidate.is_file() and any(
                token in candidate.name.lower() for token in ("partial", "retry")
            ):
                no_partial_retry = False
        entry_result = {
            "directory": str(local),
            "source_directory": str(directory),
            "manifest_completed": manifest.get("status") == "completed",
            "primary_count": entry.get("primary_count"),
            "audit_primary_count": entry.get("primary_count"),
            "runtime_attestation_present": (local / "runtime_attestation.json").is_file(),
            "code_hashes_fresh_against_current_tree": code_fresh,
            "classification": "excluded_stale_development_evidence",
            "eligible_for_formal_aggregation": False,
        }
        method_rows[method] = entry_result
        all_complete = all_complete and bool(
            entry_result["manifest_completed"]
            and entry_result["runtime_attestation_present"]
            and entry_result["primary_count"] == 6
        )
        any_stale = any_stale or not code_fresh
    return {
        "path": str(path),
        "audit_present": audit_path.is_file(),
        "methods": method_rows,
        "complete_primary_ledgers": all_complete,
        "stale_code_is_explicitly_excluded": any_stale,
        "no_partial_or_retry_files": no_partial_retry,
        "eligible_for_formal_aggregation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--version", default="v8")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "g0_evidence_audit_v8.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load(protocol_path)
    pool_summary = load(POOL_SUMMARY)
    pool_rows = count_rows(POOL)
    test_probe = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = 0
    for line in test_probe.stdout.splitlines():
        if " tests collected" in line:
            try:
                collected = int(line.strip().split()[0])
            except (ValueError, IndexError):
                pass
    active_protocol_sha = sha256(protocol_path)
    active_formal = []
    for manifest_path in RESULTS.rglob("run_manifest.json"):
        manifest = load(manifest_path)
        identity = manifest.get("run_identity", {})
        if (
            identity.get("stage") == "formal"
            and manifest.get("status") == "completed"
            and manifest.get("hashes", {}).get("protocol_sha256") == active_protocol_sha
        ):
            active_formal.append(str(manifest_path))
    old_active_paths = (
        ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v2.json",
        ROOT / "experiments/results/g7_mbo_gate_afe_v2.json",
        ROOT / "experiments/results/g7_mbo_formal_afe_v2_20260805",
    )
    checks = {
        "old_failed_versions_not_active": all(
            not path.exists() for path in old_active_paths
        ),
        "active_protocol_matches_version": protocol.get("schema_version")
        == f"radp_search_afe_matched_qwen235b_protocol.{args.version}",
        "pool_row_count_is_50000": pool_rows == 50000,
        "pool_metadata_matches_rows": pool_summary.get("sample_size") == pool_rows
        and pool_summary.get("hidden_sample_recovered") is not True,
        "test_collection_succeeded": test_probe.returncode == 0 and collected > 0,
        "no_completed_formal_matrix": not active_formal,
    }
    result = {
        "schema_version": f"g0_evidence_audit.{args.version}",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "cleanup_trash": str(TRASH),
        "cleanup_trash_is_scientific_input": False,
        "pool": {
            "path": str(POOL),
            "rows": pool_rows,
            "summary": pool_summary,
            "sha256": sha256(POOL),
        },
        "tests": {
            "collected": collected,
            "pytest_returncode": test_probe.returncode,
            "stdout_tail": test_probe.stdout.splitlines()[-3:],
            "stderr_tail": test_probe.stderr.splitlines()[-3:],
        },
        "active_completed_formal_manifests": active_formal,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": "G0 evidence/provenance only; no API authorization and no performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": result["passed"], "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
