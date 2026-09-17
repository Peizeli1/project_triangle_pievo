#!/usr/bin/env python3
"""Audit whether the released PiEvo non-MBO surrogate is runnable locally.

This is deliberately a provenance and smoke test.  It does not fabricate an
official SPO candidate pool or leaderboard protocol when those artifacts are
not present in the local release.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PIEVO = ROOT / "pievo"
SPO_ROOT = PIEVO / "surrogates/envs/Supercon"


def main() -> None:
    model_path = SPO_ROOT / "models/best_supercon_model.pth"
    processor_path = SPO_ROOT / "models/supercon_processor.pkl"
    config_path = SPO_ROOT / "../config.yaml"

    # The release exposes a surrogate smoke interface, but no official SPO
    # task pool/evaluator alongside it.  Keep these checks explicit so an
    # eventual official adapter can be added without relabelling this audit.
    task_candidates = [
        PIEVO / "tasks",
        SPO_ROOT / "data",
        SPO_ROOT / "task.yaml",
        SPO_ROOT / "benchmark.yaml",
    ]
    task_artifacts = [str(path) for path in task_candidates if path.exists()]

    predictions: dict[str, float | None] = {}
    smoke_error: str | None = None
    try:
        warnings.filterwarnings("ignore", category=UserWarning)
        sys.path.insert(0, str(PIEVO / "surrogates/envs"))
        from Supercon.predict import Inference

        inference = Inference(str(model_path), 470)
        for formula in ("Y1Ba2Cu3O7", "MgB2"):
            value = inference.predict_tc_from_formula(formula)
            predictions[formula] = float(value) if value is not None else None
    except Exception as exc:  # pragma: no cover - environment-specific failure path
        smoke_error = f"{type(exc).__name__}: {exc}"

    checks = {
        "spo_surrogate_model_present": model_path.is_file(),
        "spo_processor_present": processor_path.is_file(),
        "spo_config_present": config_path.is_file(),
        "surrogate_smoke_returned_two_finite_predictions": (
            len(predictions) == 2 and all(value is not None for value in predictions.values())
        ),
        "official_spo_task_pool_present": bool(task_artifacts),
    }
    result = {
        "audit_type": "released_non_mbo_surrogate_adapter_readiness",
        "model": "qwen3-235b-a22b-instruct-2507",
        "seed": 0,
        "task": "SPO",
        "released_component": "Supercon surrogate exposed by the local PiEvo release",
        "model_path": str(model_path),
        "processor_path": str(processor_path),
        "config_path": str(config_path),
        "task_artifacts_found": task_artifacts,
        "surrogate_predictions": predictions,
        "smoke_error": smoke_error,
        "checks": checks,
        "all_checks_pass": all(checks.values())
        and not smoke_error,
        "official_run_status": "not_run_no_official_task_pool_or_evaluator_in_local_release",
        "boundary": (
            "The released SPO/Supercon surrogate is locally loadable and passes a two-input smoke test; "
            "this is not an official SPO benchmark run and must not be reported as one."
        ),
    }
    output = EXP / "results/official_task_adapter_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "surrogate_smoke": checks["surrogate_smoke_returned_two_finite_predictions"],
        "official_spo_task_pool_present": checks["official_spo_task_pool_present"],
        "official_run_status": result["official_run_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
