from __future__ import annotations

import math

import numpy as np

from experiments.compute_gemini_apd import apd_percent
from experiments.generate_gemini_table1_report import aggregate, table2


def test_apd_uses_normalized_euclidean_percentage() -> None:
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    expected = (math.sqrt(2.0) + 0.0 + math.sqrt(2.0)) / 3.0 * 100.0
    assert math.isclose(apd_percent(vectors), expected)


def test_apd_is_aggregated_over_all_three_seeds_and_rendered() -> None:
    cells = [
        {
            "method": "radp",
            "task": "MBO",
            "seed": seed,
            "complete": True,
            "sq_percent": 10.0,
            "auoc_percent": 20.0,
            "apd_percent": 30.0 + seed,
            "total_evaluator_actions": 24,
            "failure": None,
        }
        for seed in (0, 1, 2)
    ]
    item = aggregate(cells, "radp", "MBO")
    assert item["apd_claimable"] is True
    assert math.isclose(item["apd_percent"]["mean"], 31.0)
    assert "31.00 +/- 1.00" in table2([item])
