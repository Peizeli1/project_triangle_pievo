#!/usr/bin/env python3
"""Evaluate the released PiFlow SPO data and surrogate without an LLM call.

This adapter uses the data, processor, and checkpoint shipped by the official
PiFlow repository.  It is deliberately separate from a paper leaderboard run:
the release does not contain the authors' agent trajectories or a benchmark
runner.  The output therefore records an official-data/official-surrogate
evaluation and the exact six-system filtering rule from the PiEvo appendix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "official_sources/PiFlow"
DEFAULT_OUTPUT = ROOT / "experiments/results/official_piflow_spo_seed0"
PIEVO_REFERENCE = 298.5

SYSTEMS: dict[str, set[str]] = {
    "La-Sr-Cu-O": {"La", "Sr", "Cu", "O"},
    "La-Ba-Cu-O": {"La", "Ba", "Cu", "O"},
    "Y-Ba-Cu-O": {"Y", "Ba", "Cu", "O"},
    "Bi-Sr-Ca-Cu-O": {"Bi", "Sr", "Ca", "Cu", "O"},
    "Tl-Ba-Ca-Cu-O": {"Tl", "Ba", "Ca", "Cu", "O"},
    "Hg-Ba-Ca-Cu-O": {"Hg", "Ba", "Ca", "Cu", "O"},
}
FORMULA_RE = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def formula_elements(formula: Any) -> set[str]:
    if formula is None or (isinstance(formula, float) and math.isnan(formula)):
        return set()
    text = str(formula).replace("-", "")
    return {element for element, _ in FORMULA_RE.findall(text)}


def system_matches(formula: Any) -> list[str]:
    elements = formula_elements(formula)
    if "Cu" not in elements or "O" not in elements:
        return []
    # PiEvo states that the formula may use the elements of a listed system;
    # subset matching therefore keeps stoichiometric variants such as La2CuO4
    # inside La-Sr-Cu-O while excluding unrelated elements.
    return [name for name, allowed in SYSTEMS.items() if elements <= allowed]


def load_official_predictor(repo: Path):
    env_root = repo / "envs/AgenX_Supercon"
    sys.path.insert(0, str(env_root))
    from src.data_processor import SuperconDataProcessor  # type: ignore
    from src.model import TcPredictor  # type: ignore

    processor_path = env_root / "models/supercon_processor.pkl"
    model_path = env_root / "models/best_supercon_model.pth"
    with processor_path.open("rb") as handle:
        processor_data = pickle.load(handle)
    model = TcPredictor(input_size=processor_data["n_features"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return SuperconDataProcessor(), processor_data, model, model_path, processor_path


def predict_dataframe(repo: Path, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    processor, processor_data, model, model_path, processor_path = load_official_predictor(repo)
    # PiFlow's released processor reads its pickle through ./models.  Run it
    # under the official module directory so the shipped implementation is
    # used verbatim rather than reimplementing feature extraction here.
    old_cwd = Path.cwd()
    os.chdir(repo / "envs/AgenX_Supercon")
    try:
        X = processor.process_input(data[["element", "str3"]].copy())
    finally:
        os.chdir(old_cwd)
    with torch.no_grad():
        predictions = model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy().reshape(-1)
    metadata = {
        "model_path": str(model_path),
        "processor_path": str(processor_path),
        "input_size": int(processor_data["n_features"]),
        "model_sha256": sha256(model_path),
        "processor_sha256": sha256(processor_path),
    }
    return predictions, metadata


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    observed_idx = group["tc"].idxmax()
    predicted_idx = group["predicted_tc"].idxmax()
    observed = group.loc[observed_idx]
    predicted = group.loc[predicted_idx]
    return {
        "rows": int(len(group)),
        "unique_formulas": int(group["element"].nunique()),
        "observed_max_tc": float(group["tc"].max()),
        "observed_max_formula": str(observed["element"]),
        "predicted_max_tc": float(group["predicted_tc"].max()),
        "predicted_max_formula": str(predicted["element"]),
        "predicted_solution_quality_percent": float(
            100.0 * group["predicted_tc"].max() / PIEVO_REFERENCE
        ),
        "predicted_top5": [
            {
                "formula": str(row["element"]),
                "structure": None if pd.isna(row["str3"]) else str(row["str3"]),
                "predicted_tc": float(row["predicted_tc"]),
                "observed_tc": float(row["tc"]),
            }
            for _, row in group.sort_values("predicted_tc", ascending=False).head(5).iterrows()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_REPO / "envs/AgenX_Supercon/data/superconductor_data.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data, sep="\t")
    clean = raw.dropna(subset=["element", "tc"]).copy().reset_index(drop=True)
    predictions, model_meta = predict_dataframe(args.repo, clean)
    clean["predicted_tc"] = predictions
    clean["system_matches"] = clean["element"].map(system_matches)
    clean.to_csv(args.output_dir / "official_spo_row_evaluations.csv", index=False)

    system_rows: dict[str, Any] = {}
    for name in SYSTEMS:
        rows = clean[clean["system_matches"].map(lambda values: name in values)]
        system_rows[name] = summarize_group(rows) if len(rows) else {"rows": 0}

    all_summary = summarize_group(clean)
    checks = {
        "official_data_present": args.data.is_file(),
        "official_model_present": Path(model_meta["model_path"]).is_file(),
        "official_processor_present": Path(model_meta["processor_path"]).is_file(),
        "expected_row_count": len(raw) == 26321,
        "expected_feature_count": model_meta["input_size"] == 509,
        "all_predictions_finite": bool(np.isfinite(predictions).all()),
        "all_six_systems_have_rows": all(item["rows"] > 0 for item in system_rows.values()),
    }
    result = {
        "audit_type": "official_piflow_spo_data_and_surrogate_evaluation",
        "source": {
            "repository": "https://github.com/amair-lab/PiFlow",
            "repository_commit": git_commit(args.repo),
            "paper": "https://arxiv.org/abs/2505.15047",
            "pievo_paper": "https://arxiv.org/abs/2602.06448",
            "data_path": str(args.data),
            "data_sha256": sha256(args.data),
        },
        "model": "qwen3-235b-a22b-instruct-2507",
        "seed": args.seed,
        "llm_calls": 0,
        "reference_tc_kelvin": PIEVO_REFERENCE,
        "raw_dataset": {
            "rows": int(len(raw)),
            "clean_rows": int(len(clean)),
            "unique_formulas": int(clean["element"].nunique()),
            "unique_structure_types": int(clean["str3"].nunique(dropna=True)),
            "observed_max_tc": float(clean["tc"].max()),
        },
        "official_model": model_meta,
        "all_data_summary": all_summary,
        "six_system_summaries": system_rows,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "boundary": (
            "This is an official-data/official-surrogate evaluation. The released repositories do not include "
            "the authors' original agent trajectories or a leaderboard runner, so it is not an official PiEvo "
            "leaderboard reproduction."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir / "summary.json"), "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
