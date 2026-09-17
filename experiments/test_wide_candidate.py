from __future__ import annotations

import json
from pathlib import Path

import pytest

from pievo.radp.wide_candidate import (
    load_unlabeled_smiles,
    broad_structural_portfolio,
    stream_fingerprint,
    structural_portfolio,
)


def test_loader_reads_only_canonical_smiles_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "pool.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"canonical_smiles": "C1=CC=CC=C1", "pchembl": 999.0}),
                json.dumps({"canonical_smiles": "c1ccccc1", "pchembl": -999.0}),
                json.dumps({"canonical_smiles": "CCO", "pchembl": 12.0}),
                json.dumps({"canonical_smiles": "not a smiles [", "pchembl": 1000.0}),
                json.dumps({"pchembl": 1000.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_unlabeled_smiles(source) == ["c1ccccc1", "CCO"]

    # Changing labels must not change the candidate stream or its fingerprint.
    relabeled = tmp_path / "relabeled.jsonl"
    relabeled.write_text(
        "\n".join(
            [
                json.dumps({"canonical_smiles": "C1=CC=CC=C1", "pchembl": -12345.0}),
                json.dumps({"canonical_smiles": "c1ccccc1", "pchembl": 12345.0}),
                json.dumps({"canonical_smiles": "CCO", "pchembl": -54321.0}),
                json.dumps({"canonical_smiles": "not a smiles [", "pchembl": -1000.0}),
                json.dumps({"pchembl": -1000.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_unlabeled_smiles(relabeled) == load_unlabeled_smiles(source)
    assert stream_fingerprint(load_unlabeled_smiles(relabeled)) == stream_fingerprint(
        load_unlabeled_smiles(source)
    )


def test_structural_portfolio_is_deterministic_valid_unique_and_excludes() -> None:
    candidates = [
        "c1ccccc1",
        "C1=CC=CC=C1",
        "CCO",
        "C1CCCCCCC1",
        "C1C2CC3CC1CC(C2)C3",
        "O=C(O)c1ccccc1",
    ]
    first = structural_portfolio(candidates, budget=4, seed=17)
    second = structural_portfolio(list(reversed(candidates)), budget=4, seed=17)

    assert first == second
    assert len(first) == len(set(first)) == 4
    assert all(first)
    assert "c1ccccc1" not in structural_portfolio(
        candidates, budget=4, seed=17, excluded=["C1=CC=CC=C1"]
    )
    assert stream_fingerprint(first) == stream_fingerprint(second)

    # Independent seeds must produce independent candidate streams.  A prior
    # implementation sorted every scaffold after shuffling and silently
    # returned the same stream for every seed.
    third = structural_portfolio(candidates, budget=4, seed=18)
    assert stream_fingerprint(third) != stream_fingerprint(first)


def test_structural_portfolio_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        structural_portfolio(["CCO"], budget=0, seed=0)


def test_broad_structural_portfolio_is_seed_robust_and_label_blind() -> None:
    candidates = [
        "CCOc1ccc2ccccc2c1",
        "CCN(CC)CCOC(=O)c1ccc(Cl)cc1",
        "C1CCCCCCC1",
        "CC(C)C(=O)N1CCN(c2ccccc2)CC1",
        "C" * 1,
    ]
    first = broad_structural_portfolio(candidates, budget=4, seed=0)
    second = broad_structural_portfolio(candidates, budget=4, seed=1)

    assert len(first) == len(set(first))
    assert len(second) == len(set(second))
    assert first != second
    assert all("." not in candidate and "+" not in candidate for candidate in first + second)


def test_broad_structural_portfolio_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        broad_structural_portfolio(["CCO"], budget=0, seed=0)
