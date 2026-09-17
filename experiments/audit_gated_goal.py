#!/usr/bin/env python3
"""Fail-closed reducer for the staged RADP-Search goal.

This reducer never upgrades a diagnostic into a formal claim.  It only
authorizes the next stage when every preceding artifact is present, fresh and
passed.  The output is intentionally kept outside the project result tree by
the shell orchestrator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def audit_screen(path: Path, expected_seed: int) -> dict[str, Any]:
    artifact = read_json(path)
    methods = artifact.get("methods", {})
    if not isinstance(methods, dict):
        methods = {}
    checks = artifact.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    return {
        "path": str(path.resolve()),
        "present": path.is_file(),
        "passed": artifact.get("passed") is True,
        "schema_version": artifact.get("schema_version"),
        "task": artifact.get("task"),
        "seed": artifact.get("seed"),
        "budget": artifact.get("budget"),
        "mode": artifact.get("mode"),
        "expected_seed": expected_seed,
        "method_names": sorted(methods),
        "checks": checks,
        "method_primary_counts": {
            name: value.get("primary_count")
            for name, value in methods.items()
            if isinstance(value, dict)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed0 = audit_screen(
        args.root / "seed0" / "official_mbo_development_screen_audit.json",
        0,
    )
    seed1 = audit_screen(
        args.root / "seed1" / "official_mbo_development_screen_audit.json",
        1,
    )
    checks = {
        "seed0_artifact_passed": seed0["passed"],
        "seed1_artifact_passed": seed1["passed"],
        "both_are_official_mbo": seed0["task"] == "MBO" and seed1["task"] == "MBO",
        "both_are_source_confusion_diagnostics": (
            seed0["mode"] == "source_confusion"
            and seed1["mode"] == "source_confusion"
        ),
        "seeds_are_0_and_1": seed0["seed"] == 0 and seed1["seed"] == 1,
        "both_use_six_primary_budget": seed0["budget"] == 6 and seed1["budget"] == 6,
        "both_have_exactly_six_primary_per_method": all(
            set(item["method_primary_counts"].values()) == {6}
            and set(item["method_names"]) == {"pievo", "radp"}
            for item in (seed0, seed1)
        ),
    }
    result = {
        "schema_version": "radp_search_gated_goal.v1",
        "claim_boundary": (
            "Only a controlled official-MBO development diagnostic is allowed; "
            "formal, paper-setting and external claims remain locked."
        ),
        "stage": "seed0_then_seed1",
        "screens": {"seed0": seed0, "seed1": seed1},
        "checks": checks,
        "passed": all(checks.values()),
        "next_stage": "independent_verifier_and_formal_readiness"
        if all(checks.values())
        else "stop",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 12


if __name__ == "__main__":
    raise SystemExit(main())
