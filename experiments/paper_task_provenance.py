#!/usr/bin/env python3
"""Audit task definitions recoverable from the PiEvo paper and release.

This is a provenance check, not an assertion that the paper's hidden runtime
or leaderboard evaluator has been recovered.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "official_sources/PiEvo/assets/2602.06448v1.pdf"
MATCHED_REPO = ROOT / "official_sources/PiEvo_matched_qwen235b"
OUTPUT = ROOT / "experiments/results/paper_task_provenance.json"


def extract_pdf_text() -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(PAPER), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return result.stdout or "", result.stderr.strip() or f"pdftotext exit {result.returncode}"
    return result.stdout, None


def task_definition(text: str, required_phrases: list[str]) -> dict[str, Any]:
    present = {phrase: phrase.lower() in text.lower() for phrase in required_phrases}
    return {"required_phrases": present, "all_present": all(present.values())}


def release_tree() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(ROOT / "official_sources/PiEvo"), "ls-tree", "-r", "--name-only", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [line for line in output.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return []


def main() -> int:
    text, pdf_error = extract_pdf_text()
    sys.path.insert(0, str(MATCHED_REPO / "surrogates/envs"))
    try:
        from TMC.src.ligands import LIGAND_POOL  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific import
        LIGAND_POOL = {}
        tmc_import_error = f"{type(exc).__name__}: {exc}"
    else:
        tmc_import_error = None

    tmc_smiles = [str(value.get("smiles")) for value in LIGAND_POOL.values()]
    tmc_smiles_in_paper = sum(smiles in text for smiles in tmc_smiles)
    tree = release_tree()
    runner_like_files = [
        path for path in tree
        if any(token in path.lower() for token in ("runner", "leaderboard", "benchmark", "metric"))
    ]
    surrogate_evaluator_files = [path for path in tree if "evaluator" in path.lower()]
    task_results = {
        "NHO": {
            "paper_definition": task_definition(text, ["Nanohelix Optimization", "fiber-radius", "wavelength", "g-factor"]),
            "released_surrogate": (MATCHED_REPO / "surrogates/envs/Nanohelix/models/nanohelix_torch_model.pth").is_file(),
            "task_sampler_recovered": False,
            "official_runner_recovered": False,
            "boundary": "Paper bounds/format are recoverable; the paper's sampling trajectory and official runner are not published in the release.",
        },
        "MBO": {
            "paper_definition": task_definition(text, ["Molecular Bio-activity Optimization", "pChEMBL"]),
            "released_surrogate": (MATCHED_REPO / "surrogates/envs/ChEMBL35/models/best_r2_model.pt").is_file(),
            "hidden_50000_sample_recovered": False,
            "task_sampler_recovered": False,
            "official_runner_recovered": False,
            "boundary": "The paper says it follows PiFlow and describes a 50,000-molecule sample, but does not publish row IDs/seed; local pool is a deterministic reconstruction.",
        },
        "SPO": {
            "paper_definition": task_definition(text, ["Superconductor Optimization", "La-Sr-Cu-O System", "Hg-Ba-Ca-Cu-O System"]),
            "released_surrogate": (MATCHED_REPO / "surrogates/envs/Supercon/models/best_supercon_model.pth").is_file(),
            "task_sampler_recovered": False,
            "official_runner_recovered": False,
            "boundary": "Paper systems and formula constraints are recoverable; no official candidate sampler/leaderboard evaluator is released.",
        },
        "TMC": {
            "paper_definition": task_definition(text, ["Transition Metal Complex Optimization", "Table 9", "1367499 records"]),
            "released_surrogate": (MATCHED_REPO / "surrogates/envs/TMC/models/best_tmc_model.pth").is_file(),
            "paper_table9_ligand_count": len(tmc_smiles),
            "paper_table9_smiles_present": tmc_smiles_in_paper == len(tmc_smiles) and len(tmc_smiles) == 50,
            "paper_table9_smiles_present_count": tmc_smiles_in_paper,
            "task_sampler_recovered": False,
            "official_runner_recovered": False,
            "boundary": "The 50-ligand SMILES pool in paper Table 9 matches the released surrogate's ligand metadata; the original 1,367,499-row dataset/evaluator is not bundled.",
        },
    }
    for task in task_results.values():
        task["paper_definition_provenance_ok"] = task["paper_definition"]["all_present"]

    reconstruction_ready = all(
        value.get("paper_definition_provenance_ok") and value.get("released_surrogate")
        for value in task_results.values()
    ) and task_results["TMC"].get("paper_table9_smiles_present") is True
    official_ready = all(
        value.get("task_sampler_recovered") and value.get("official_runner_recovered")
        for value in task_results.values()
    )
    result = {
        "schema_version": "paper_task_provenance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_source": str(PAPER.resolve()),
        "paper_text_extraction_error": pdf_error,
        "tasks": task_results,
        "official_release": {
            "git_tree_file_count": len(tree),
            "runner_like_files": runner_like_files,
            "surrogate_evaluator_files": surrogate_evaluator_files,
            "official_task_runner_evaluator_present": bool(runner_like_files),
        },
        "tmc_import_error": tmc_import_error,
        "paper_faithful_reconstruction_ready": reconstruction_ready,
        "official_formal_assets_ready": official_ready,
        "claim_boundary": {
            "paper_faithful_reconstruction": "development/transfer evidence only until an official runner/evaluator is independently verified",
            "official_leaderboard_reproduction": "not supported by this artifact",
            "qwen3_32b_paper_sota": "requires exact paper-setting model and runtime track",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "paper_faithful_reconstruction_ready": reconstruction_ready,
        "official_formal_assets_ready": official_ready,
        "tmc_table9_smiles_present_count": tmc_smiles_in_paper,
        "release_runner_like_files": runner_like_files,
    }, ensure_ascii=False, indent=2))
    return 0 if reconstruction_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
