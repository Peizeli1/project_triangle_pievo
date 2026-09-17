#!/usr/bin/env python3
"""Offline readiness audit for the four PiEvo task components.

This distinguishes a loadable released surrogate from an official benchmark
runner/task pool.  It never calls an LLM or relabels a surrogate smoke test as
an official PiEvo result.
"""

from __future__ import annotations

import argparse
import json
import math
import importlib
import os
import sys
import warnings
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "official_sources/PiEvo_matched_qwen235b"
ENV_ROOT = REPO / "surrogates/envs"
RESULTS = ROOT / "experiments/results"
MBO_POOL = RESULTS / "official_chembl35_pool_seed0/chembl35_pool_seed0.jsonl"
MBO_POOL_SUMMARY = RESULTS / "official_chembl35_pool_seed0/chembl35_pool_seed0_summary.json"
PAPER_PROVENANCE = RESULTS / "paper_task_provenance.json"
RUNNER = ROOT / "experiments/run_matched_qwen235b_task.sh"
CONFIGS = {
    "NHO": ("official_pievo_nho_task.yaml", "official_pievo_nho_model.yaml"),
    "MBO": ("official_pievo_mbo_task.yaml", "official_pievo_mbo_model_short_smiles.yaml"),
    "SPO": ("official_pievo_spo_task.yaml", "official_pievo_spo_model.yaml"),
    "TMC": ("official_pievo_tmc_task.yaml", "official_pievo_tmc_model.yaml"),
}
ADAPTER = REPO / "pievo/tools/tool_scientific_official.py"


def config_text(task: str) -> str:
    """Return the frozen task description used by the matched evaluator audit."""
    task_path = ROOT / "experiments/configs" / CONFIGS[task][0]
    try:
        return task_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def candidate_constraints_verified(task: str) -> tuple[bool, str]:
    """Verify the input semantics needed for a released-evaluator comparison.

    These checks intentionally do not assert that the hidden paper sampler was
    recovered.  They only establish that the candidate domain described in the
    frozen adapter is explicit and auditable.
    """
    text = config_text(task)
    required = {
        "NHO": ("continuous [20, 60]", "x_y_z", "direction"),
        "MBO": ("valid molecular structures", "valid-SMILES", "single line"),
        "SPO": ("La-Sr-Cu-O", "Hg-Ba-Ca-Cu-O", "exact formula string"),
        "TMC": ("Released ligand pool", "charge -1", "sum to -2"),
    }[task]
    missing = [phrase for phrase in required if phrase not in text]
    return (not missing, "all frozen candidate constraints present" if not missing else f"missing: {missing}")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def smoke_nho() -> dict[str, Any]:
    sys.path.insert(0, str(ENV_ROOT))
    try:
        from Nanohelix.src.models import predict_g_factor

        result = predict_g_factor(
            {
                "fiber_radius": 30.0,
                "helix_radius": 60.0,
                "n_turns": 6.0,
                "pitch": 120.0,
                "wavelength": 450.0,
                "x_y": 2,
                "direction": 0,
            }
        )
        return {"ok": finite(result.get("g_factor")), "output": result.get("g_factor")}
    except Exception as exc:  # pragma: no cover - environment-specific model loading
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def smoke_mbo() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(ENV_ROOT))
        from ChEMBL35.predict import MoleculePredictor

        model_path = ENV_ROOT / "ChEMBL35/models/best_r2_model.pt"
        value = MoleculePredictor(str(model_path), device="cpu").predict("c1ccccc1")
        return {"ok": finite(value), "output": value}
    except Exception as exc:  # pragma: no cover - environment-specific model loading
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def smoke_spo() -> dict[str, Any]:
    try:
        warnings.filterwarnings("ignore", category=UserWarning)
        sys.path.insert(0, str(ENV_ROOT))
        from Supercon.predict import Inference

        model_path = ENV_ROOT / "Supercon/models/best_supercon_model.pth"
        value = Inference(str(model_path), 470).predict_tc_from_formula("Y1Ba2Cu3O7")
        return {"ok": finite(value), "output": value}
    except Exception as exc:  # pragma: no cover - environment-specific model loading
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def smoke_tmc() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(ENV_ROOT))
        from TMC.predict import find_tmc_value
        from TMC.src.ligands import LIGAND_POOL

        neutral = next(key for key, value in LIGAND_POOL.items() if value["charge"] == 0)
        anions = [key for key, value in LIGAND_POOL.items() if value["charge"] == -1][:2]
        candidate = "Pd_" + "_".join([anions[0], anions[1], neutral, neutral])
        result = find_tmc_value(candidate)
        return {
            "ok": result.get("success") is True and finite(result.get("output")),
            "output": result.get("output"),
            "candidate": candidate,
            "ligand_pool_size": len(LIGAND_POOL),
        }
    except Exception as exc:  # pragma: no cover - environment-specific model loading
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def smoke_adapter_registry() -> dict[str, Any]:
    """Load the task adapters without making an LLM/API call."""
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import pievo.tools as tools

        tools.discover_and_load_tools()
        required = {
            "evaluate_nanohelix",
            "evaluate_supercon",
            "evaluate_tmc",
            "verify_nanohelix",
            "verify_supercon",
            "verify_tmc",
        }
        present = set(tools.get_all_tools())
        return {
            "ok": required <= present,
            "required": sorted(required),
            "missing": sorted(required - present),
        }
    except Exception as exc:  # pragma: no cover - environment-specific imports
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def reconstruction_pool_metadata() -> dict[str, Any]:
    """Report the local MBO reconstruction without upgrading its claim scope."""

    row_count = 0
    if MBO_POOL.is_file():
        with MBO_POOL.open("r", encoding="utf-8") as handle:
            row_count = sum(1 for line in handle if line.strip())
    summary: dict[str, Any] = {}
    if MBO_POOL_SUMMARY.is_file():
        try:
            value = json.loads(MBO_POOL_SUMMARY.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                summary = value
        except (OSError, ValueError):
            summary = {}
    declared_size = summary.get("sample_size")
    return {
        "path": str(MBO_POOL),
        "present": MBO_POOL.is_file(),
        "row_count": row_count,
        "declared_sample_size": declared_size,
        "row_count_matches_declared_size": (
            isinstance(declared_size, int) and row_count == declared_size
        ),
        "sample_seed": summary.get("sample_seed"),
        "official_hidden_pool_recovered": False,
        "claim_boundary": (
            "Deterministic local ChEMBL35 reconstruction only; it does not recover "
            "the paper's hidden 50,000-molecule sample or official sampler."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v11")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config_root = ROOT / "experiments/configs"
    adapter_smoke = smoke_adapter_registry()
    paper_provenance: dict[str, Any] = {}
    if PAPER_PROVENANCE.is_file():
        try:
            paper_provenance = json.loads(PAPER_PROVENANCE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            paper_provenance = {}
    paper_tasks = paper_provenance.get("tasks", {})
    mbo_pool = reconstruction_pool_metadata()
    tasks = {
        "NHO": {
            "released_surrogate_present": (ENV_ROOT / "Nanohelix/models/nanohelix_torch_model.pth").is_file(),
            "official_task_pool_present": False,
            "official_runner_present": False,
            "matched_adapter_present": ADAPTER.is_file(),
            "matched_config_present": all((config_root / name).is_file() for name in CONFIGS["NHO"]),
            "paper_definition_provenance_ok": paper_tasks.get("NHO", {}).get("paper_definition_provenance_ok", False),
            "candidate_constraints_verified": candidate_constraints_verified("NHO")[0],
            "candidate_constraints_detail": candidate_constraints_verified("NHO")[1],
            "smoke": smoke_nho(),
            "boundary": "Released 7-feature-compatible surrogate component; no official task sampler/runner in local release.",
        },
        "MBO": {
            "released_surrogate_present": (ENV_ROOT / "ChEMBL35/models/best_r2_model.pt").is_file(),
            "official_task_pool_present": False,
            "official_runner_present": False,
            "matched_adapter_present": (REPO / "pievo/tools/tool_chembl_official.py").is_file(),
            "matched_config_present": all((config_root / name).is_file() for name in CONFIGS["MBO"]),
            "controlled_local_pool_present": mbo_pool["present"],
            "controlled_local_pool": mbo_pool,
            "paper_definition_provenance_ok": paper_tasks.get("MBO", {}).get("paper_definition_provenance_ok", False),
            "hidden_50000_sample_recovered": paper_tasks.get("MBO", {}).get("hidden_50000_sample_recovered", False),
            "candidate_constraints_verified": candidate_constraints_verified("MBO")[0],
            "candidate_constraints_detail": candidate_constraints_verified("MBO")[1],
            "smoke": smoke_mbo(),
            "boundary": mbo_pool["claim_boundary"],
        },
        "SPO": {
            "released_surrogate_present": (ENV_ROOT / "Supercon/models/best_supercon_model.pth").is_file(),
            "official_task_pool_present": False,
            "official_runner_present": False,
            "matched_adapter_present": ADAPTER.is_file(),
            "matched_config_present": all((config_root / name).is_file() for name in CONFIGS["SPO"]),
            "paper_definition_provenance_ok": paper_tasks.get("SPO", {}).get("paper_definition_provenance_ok", False),
            "candidate_constraints_verified": candidate_constraints_verified("SPO")[0],
            "candidate_constraints_detail": candidate_constraints_verified("SPO")[1],
            "smoke": smoke_spo(),
            "boundary": "Released Supercon surrogate is loadable; official candidate pool/evaluator is absent.",
        },
        "TMC": {
            "released_surrogate_present": (ENV_ROOT / "TMC/models/best_tmc_model.pth").is_file(),
            "official_task_pool_present": False,
            "official_runner_present": False,
            "matched_adapter_present": ADAPTER.is_file(),
            "matched_config_present": all((config_root / name).is_file() for name in CONFIGS["TMC"]),
            "released_ligand_pool_present": (ENV_ROOT / "TMC/src/ligands.py").is_file(),
            "paper_definition_provenance_ok": paper_tasks.get("TMC", {}).get("paper_definition_provenance_ok", False),
            "paper_table9_smiles_present": paper_tasks.get("TMC", {}).get("paper_table9_smiles_present", False),
            "candidate_constraints_verified": candidate_constraints_verified("TMC")[0],
            "candidate_constraints_detail": candidate_constraints_verified("TMC")[1],
            "smoke": smoke_tmc(),
            "boundary": "Released TMC surrogate and 50-ligand metadata are loadable; official task sampler/leaderboard evaluator is absent.",
        },
    }
    for task in tasks.values():
        task["released_evaluator_verified"] = bool(
            task.get("released_surrogate_present")
            and task.get("smoke", {}).get("ok") is True
        )
        task["matched_evaluator_assets_verified"] = bool(
            task.get("paper_definition_provenance_ok") is True
            and task.get("candidate_constraints_verified") is True
            and task.get("released_evaluator_verified") is True
            and task.get("matched_adapter_present") is True
            and task.get("matched_config_present") is True
        )
    result = {
        "schema_version": f"official_four_task_readiness.{args.version}",
        "model_track": "matched_qwen235b",
        "seed": 0,
        "tasks": tasks,
        "matched_reconstruction": {
            "runner_present": RUNNER.is_file() and os.access(RUNNER, os.X_OK),
            "adapter_registry_smoke": adapter_smoke,
            "all_task_configs_present": all(task["matched_config_present"] for task in tasks.values()),
            "all_surrogate_smoke_pass": all(task["smoke"].get("ok") is True for task in tasks.values()),
            "ready": (
                RUNNER.is_file()
                and os.access(RUNNER, os.X_OK)
                and adapter_smoke.get("ok") is True
                and all(task["matched_adapter_present"] and task["matched_config_present"] for task in tasks.values())
                and all(task["smoke"].get("ok") is True for task in tasks.values())
            ),
            "claim_boundary": "Matched reconstruction only; it is not an official PiEvo task pool/leaderboard reproduction.",
        },
        "matched_evaluator_assets": {
            "ready": bool(
                adapter_smoke.get("ok") is True
                and all(task.get("matched_evaluator_assets_verified") is True for task in tasks.values())
            ),
            "tasks": {
                name: {
                    "paper_definition_provenance_ok": task.get("paper_definition_provenance_ok"),
                    "candidate_constraints_verified": task.get("candidate_constraints_verified"),
                    "released_evaluator_verified": task.get("released_evaluator_verified"),
                    "matched_adapter_present": task.get("matched_adapter_present"),
                    "matched_config_present": task.get("matched_config_present"),
                    "hidden_paper_sampler_required": False if name == "MBO" else None,
                }
                for name, task in tasks.items()
            },
            "claim_boundary": "Formal matched comparison against released evaluators; not exact paper leaderboard reproduction.",
        },
        "paper_provenance": {
            "artifact_present": PAPER_PROVENANCE.is_file(),
            "paper_faithful_reconstruction_ready": paper_provenance.get("paper_faithful_reconstruction_ready", False),
            "official_formal_assets_ready": paper_provenance.get("official_formal_assets_ready", False),
            "source": str(PAPER_PROVENANCE),
        },
        "exact_apd_evaluator_present": False,
        "formal_matched_run_ready": bool(
            RUNNER.is_file()
            and os.access(RUNNER, os.X_OK)
            and adapter_smoke.get("ok") is True
            and all(task.get("matched_evaluator_assets_verified") is True for task in tasks.values())
        ),
        "formal_official_run_ready": all(
            task.get("official_task_pool_present") and task.get("official_runner_present")
            for task in tasks.values()
        ),
        "claim_boundary": "Matched evaluator formal runs require released evaluator and candidate-constraint verification; exact official leaderboard claims still require the separate paper-setting asset gate.",
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else RESULTS / f"official_four_task_readiness_{args.version}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "smoke_pass": {name: task["smoke"].get("ok") for name, task in tasks.items()},
        "formal_official_run_ready": result["formal_official_run_ready"],
        "formal_matched_run_ready": result["formal_matched_run_ready"],
        "exact_apd_evaluator_present": result["exact_apd_evaluator_present"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
