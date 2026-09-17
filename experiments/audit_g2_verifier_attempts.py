#!/usr/bin/env python3
"""Aggregate G2 verifier attempt status without selecting from held-out labels.

This audit is a stop-control artifact. It records existing manifests and
precomputed metrics, but never trains a model, reads raw activity labels, or
upgrades a failed attempt into a provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def attempt_row(path: Path) -> dict[str, Any]:
    summary = load_json(path / "build_summary.json")
    metrics = load_json(path / "metrics.json")
    runtime = load_json(path / "runtime_manifest.json")
    dataset = load_json(path / "dataset_manifest.json")
    passed = summary.get("passed") is True
    formal_ready = runtime.get("formal_ready") is True and passed
    return {
        "path": str(path.relative_to(ROOT)),
        "classification": "formal_candidate" if formal_ready else "excluded",
        "complete_build_summary": (path / "build_summary.json").is_file(),
        "passed": passed,
        "formal_ready": formal_ready,
        "failed_checks": summary.get("failed_checks", []),
        "sample_counts": metrics.get("sample_counts", {}),
        "consensus": metrics.get("consensus", {}),
        "test_applicability_coverage": metrics.get("test_applicability_coverage"),
        "conformal_interval_width": metrics.get("conformal_interval_width"),
        "representative_coverage": summary.get("coverage", {}).get("representative"),
        "on_policy_coverage": summary.get("coverage", {}).get("on_policy"),
        "document_overlap_count": dataset.get("document_overlap_count"),
        "molecule_overlap_count": dataset.get("molecule_overlap_count"),
        "labels_used_for_selection": metrics.get("test_labels_used_for_selection_or_thresholds"),
        "labels_used_for_training": metrics.get("heldout_labels_used_for_training"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "g2_verifier_attempt_audit.json",
    )
    args = parser.parse_args()

    paths = sorted(
        path
        for root in (RESULTS, RESULTS / "excluded")
        for path in root.glob("g2_generalized_verifier_*")
        if path.is_dir()
    )
    attempts = [attempt_row(path) for path in paths]
    result = {
        "schema_version": "g2_verifier_attempt_audit.v1",
        "api_calls": 0,
        "attempt_count": len(attempts),
        "formal_ready_count": sum(row["formal_ready"] for row in attempts),
        "all_failed_attempts_excluded": all(
            row["formal_ready"] or row["classification"] == "excluded"
            for row in attempts
        ),
        "heldout_labels_used_for_selection": any(
            row["labels_used_for_selection"] is True for row in attempts
        ),
        "heldout_labels_used_for_training": any(
            row["labels_used_for_training"] is True for row in attempts
        ),
        "attempts": attempts,
        "formal_provider_available": False,
        "claim_boundary": (
            "Attempt inventory only. No G2 provider is available and no later "
            "formal Gate is authorized by this artifact."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "attempt_count": result["attempt_count"],
                "formal_ready_count": result["formal_ready_count"],
                "api_calls": 0,
            },
            indent=2,
        )
    )
    return 0 if result["formal_ready_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
