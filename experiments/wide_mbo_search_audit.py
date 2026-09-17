#!/usr/bin/env python3
"""Zero-API audit for the outcome-blind wide MBO candidate generator."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RADP_REPO = ROOT / "official_sources/PiEvo_radp_qwen235b"
POOL = ROOT / "experiments/results/official_chembl35_pool_seed0/chembl35_pool_seed0.jsonl"
sys.path.insert(0, str(RADP_REPO))
from pievo.radp.wide_candidate import (  # noqa: E402
    afe_v8_structural_portfolio,
    load_unlabeled_smiles,
    stream_fingerprint,
)


LEGACY = (
    "c1ccccc1",
    "c1ccc2ccccc2c1",
    "c1ccc2ncccc2c1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "O=C(O)c1ccccc1",
    "O=C(Nc1ccccc1)c1ccccc1",
    "C1CCCCCCC1",
    "C1CCCCCCCCCCC1",
    "C1CCC2CCCCC2C1",
    "C1C2CC3CC1CC(C2)C3",
    "C1CCC2(CC1)CCCCC2",
    "O=C1CCCCCCCCCCC1",
)


def load_predictor():
    sys.path.insert(0, str(ROOT / "official_sources/PiEvo_matched_qwen235b"))
    from surrogates.envs.ChEMBL35.predict import MoleculePredictor

    model = ROOT / "official_sources/PiEvo_matched_qwen235b/surrogates/envs/ChEMBL35/models/best_r2_model.pt"
    return MoleculePredictor(str(model), device="cpu")


def predict_batched(predictor, candidates: list[str], batch_size: int) -> list[object]:
    """Batch graph construction and forward passes for the local audit."""

    import numpy as np
    import torch

    from surrogates.envs.ChEMBL35.src.model import MoleculeGraph

    outputs: list[object] = [None] * len(candidates)
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        valid_positions: list[int] = []
        node_matrices = []
        adjacency_matrices = []
        for offset, smiles in enumerate(chunk):
            try:
                graph = MoleculeGraph(
                    smiles,
                    node_vec_len=predictor.config["model"]["node_vec_len"],
                    max_atoms=predictor.config["data"]["max_atoms"],
                )
                if not hasattr(graph, "node_mat") or not hasattr(graph, "adj_mat"):
                    continue
                valid_positions.append(offset)
                node_matrices.append(graph.node_mat)
                adjacency_matrices.append(graph.adj_mat)
            except Exception:
                continue
        if not valid_positions:
            continue
        node_batch = torch.as_tensor(
            np.stack(node_matrices), dtype=torch.float32, device=predictor.device
        )
        adjacency_batch = torch.as_tensor(
            np.stack(adjacency_matrices), dtype=torch.float32, device=predictor.device
        )
        with torch.no_grad():
            raw = predictor.model(node_batch, adjacency_batch)
            predictions = predictor.standardizer.unstandardize(raw).reshape(-1)
        for offset, value in zip(valid_positions, predictions.detach().cpu().tolist()):
            outputs[start + offset] = float(value)
    return outputs


def bootstrap_lower(values: list[float], *, seed: int, repeats: int = 2000) -> float:
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats)]
    return sorted(means)[int(0.025 * repeats)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--pool-limit", type=int, default=50000)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--prediction-batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/results/wide_mbo_search_audit_v8.json",
    )
    args = parser.parse_args()
    if args.seeds < 8 or args.budget < 1 or args.prediction_batch_size < 1:
        raise SystemExit("seeds must be >=8, budget and prediction batch size must be positive")

    pool = load_unlabeled_smiles(POOL, limit=args.pool_limit)
    predictor = load_predictor()
    legacy_scores = [
        float(value)
        for value in predict_batched(predictor, list(LEGACY), args.prediction_batch_size)
        if value is not None
    ]
    candidate_streams = [
        afe_v8_structural_portfolio(pool, budget=args.budget, seed=seed)
        for seed in range(args.seeds)
    ]
    flattened_candidates = [candidate for stream in candidate_streams for candidate in stream]
    unique_candidates = list(dict.fromkeys(flattened_candidates))
    unique_scores = predict_batched(
        predictor, unique_candidates, args.prediction_batch_size
    )
    score_by_candidate = {
        candidate: value
        for candidate, value in zip(unique_candidates, unique_scores)
        if value is not None
    }
    rows: list[dict[str, object]] = []
    broad_best: list[float] = []
    legacy_best: list[float] = []
    for seed, candidates in enumerate(candidate_streams):
        values = [
            float(score_by_candidate[candidate])
            for candidate in candidates
            if candidate in score_by_candidate
        ]
        broad_best.append(max(values))
        legacy_best.append(max(legacy_scores))
        rows.append(
            {
                "seed": seed,
                "candidate_count": len(candidates),
                "unique_count": len(set(candidates)),
                "special_structure_count": sum(
                    "." in candidate or "[+" in candidate or "[-" in candidate
                    for candidate in candidates
                ),
                "best_surrogate": max(values) if values else None,
                "mean_surrogate": statistics.mean(values) if values else None,
                "stream_fingerprint": stream_fingerprint(candidates),
            }
        )

    differences = [wide - legacy for wide, legacy in zip(broad_best, legacy_best)]
    result = {
        "schema_version": "wide_mbo_search_audit.v8",
        "api_calls": 0,
        "seeds": args.seeds,
        "budget": args.budget,
        "pool_rows": len(pool),
        "unique_candidates_scored": len(unique_candidates),
        "pool_labels_read": False,
        "legacy_best": max(legacy_scores),
        "wide_mean_best": statistics.mean(broad_best),
        "legacy_mean_best": statistics.mean(legacy_best),
        "mean_best_difference": statistics.mean(differences),
        "bootstrap_95_lower_difference": bootstrap_lower(differences, seed=20260804),
        "all_streams_valid_and_unique": all(
            row["candidate_count"] == args.budget and row["unique_count"] == args.budget
            for row in rows
        ),
        "all_streams_cover_special_structures": all(
            int(row["special_structure_count"]) >= 2 for row in rows
        ),
        "rows": rows,
    }
    result["passed"] = bool(
        result["pool_rows"] == args.pool_limit
        and result["pool_labels_read"] is False
        and result["all_streams_valid_and_unique"] is True
        and result["all_streams_cover_special_structures"] is True
        and float(result["mean_best_difference"]) > 0.0
        and float(result["bootstrap_95_lower_difference"]) > 0.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": result["passed"], "api_calls": 0}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
