#!/usr/bin/env python3
"""Audit MBO seed variance and paired statistics without running an API."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
    "pievo",
    "radp",
)
SEEDS = (0, 1, 2)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def exact_paired_permutation(differences: np.ndarray, alternative: str) -> float:
    observed = float(np.mean(differences))
    values = np.array(
        [float(np.mean(differences * np.array(signs))) for signs in itertools.product((-1.0, 1.0), repeat=len(differences))]
    )
    if alternative == "greater":
        return float(np.mean(values >= observed - 1e-12))
    if alternative == "less":
        return float(np.mean(values <= observed + 1e-12))
    return float(np.mean(np.abs(values) >= abs(observed) - 1e-12))


def bootstrap_ci(differences: np.ndarray, seed: int = 20260809, samples: int = 100000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_result(radp: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    differences = radp - reference
    t_result = stats.ttest_rel(radp, reference)
    try:
        wilcoxon = stats.wilcoxon(differences, alternative="two-sided", method="exact")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = None
    return {
        "n": int(len(differences)),
        "differences_radp_minus_reference": differences.tolist(),
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else None,
        "bootstrap_95_percent_ci": bootstrap_ci(differences),
        "exact_paired_permutation_p_two_sided": exact_paired_permutation(differences, "two-sided"),
        "exact_paired_permutation_p_greater": exact_paired_permutation(differences, "greater"),
        "exact_sign_test_p_two_sided": float(stats.binomtest(int(np.sum(differences > 0)), len(differences), 0.5, alternative="two-sided").pvalue),
        "wilcoxon_exact_p_two_sided": wilcoxon_p,
        "paired_t_p_two_sided_descriptive": float(t_result.pvalue),
        "all_differences_positive": bool(np.all(differences > 0)),
        "significance_claimable_at_alpha_005": False if len(differences) < 6 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report_json.resolve()
    report = load_json(report_path)
    cells = [cell for cell in report.get("cells", []) if cell.get("task") == "MBO"]
    expected = len(METHODS) * len(SEEDS)
    if len(cells) != expected or any(cell.get("complete") is not True or cell.get("primary_evaluations") != 24 for cell in cells):
        raise SystemExit("MBO formal cells are incomplete")
    by_method: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    for cell in cells:
        by_method[str(cell["method"])][int(cell["seed"])] = cell
    if any(sorted(values) != list(SEEDS) for values in by_method.values()):
        raise SystemExit("MBO seed set is incomplete")

    seed_rows = []
    for method in METHODS:
        for seed in SEEDS:
            cell = by_method[method][seed]
            seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_sq_percent": float(cell["sq_percent"]),
                    "auoc_percent": float(cell["auoc_percent"]),
                    "invalid_attempts": int(cell["invalid_attempts"]),
                    "duplicate_suppressed": int(cell["duplicate_suppressed"]),
                    "total_evaluator_actions": int(cell["total_evaluator_actions"]),
                    "run_dir": cell["run_dir"],
                }
            )

    radp_sq = np.array([by_method["radp"][seed]["sq_percent"] for seed in SEEDS], dtype=float)
    comparisons = {}
    for method in METHODS:
        if method == "radp":
            continue
        reference = np.array([by_method[method][seed]["sq_percent"] for seed in SEEDS], dtype=float)
        comparisons[f"radp_vs_{method}"] = paired_result(radp_sq, reference)

    pievo_sq = np.array([by_method["pievo"][seed]["sq_percent"] for seed in SEEDS], dtype=float)
    piflow_sq = np.array([by_method["piflow"][seed]["sq_percent"] for seed in SEEDS], dtype=float)
    repeated_pievo = bool(np.allclose(pievo_sq, pievo_sq[0], rtol=0.0, atol=0.0))
    output = {
        "schema_version": "gemini_mbo_statistical_audit.v1",
        "report_json": str(report_path),
        "model": report.get("model"),
        "task": "MBO",
        "primary_metric": "seed-level best SQ percent",
        "seed_count": len(SEEDS),
        "formal_primary_per_cell": 24,
        "seed_rows": seed_rows,
        "method_summary": {
            method: {
                "mean_sq_percent": statistics.fmean([by_method[method][seed]["sq_percent"] for seed in SEEDS]),
                "std_sq_percent": statistics.stdev([by_method[method][seed]["sq_percent"] for seed in SEEDS]),
                "values_identical_across_seeds": bool(len({by_method[method][seed]["sq_percent"] for seed in SEEDS}) == 1),
            }
            for method in METHODS
        },
        "paired_comparisons": comparisons,
        "diagnostics": {
            "pievo_sq_values_identical_across_seeds": repeated_pievo,
            "pi_flow_is_current_best_mean_sq": True,
            "n3_exact_test_limit": "With n=3, the minimum two-sided exact paired permutation/sign-test p-value is 0.25; p<0.05 cannot be established.",
            "current_result_usable_for_final_significance_claim": False,
            "rerun_required": True,
            "rerun_reason": "Fix PiEvo common-generator collapse and collect at least 6 independent seeds; 8-10 seeds are preferable for stable variance and power.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Gemini MBO Statistical Audit",
        "",
        "The existing formal matrix has three seeds and 24 primary evaluations per cell.",
        "",
        "## Decision",
        "",
        "The current three-seed result is descriptive only. With n=3, an exact two-sided paired permutation/sign test cannot reach p < 0.05.",
        "A rerun is required after fixing the PiEvo common-generator collapse; collect at least six independent seeds, preferably eight to ten.",
        "",
        "## Seed-Level SQ",
        "",
        "| Method | Seed 0 | Seed 1 | Seed 2 | Mean | Std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = [by_method[method][seed]["sq_percent"] for seed in SEEDS]
        lines.append(
            f"| {method} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | "
            f"{statistics.fmean(values):.2f} | {statistics.stdev(values):.2f} |"
        )
    lines.extend(["", "## Paired RADP Comparisons", "", "| Reference | Mean delta | Bootstrap 95% CI | Exact permutation p (two-sided) |", "|---|---:|---:|---:|"])
    for method in METHODS:
        if method == "radp":
            continue
        item = comparisons[f"radp_vs_{method}"]
        ci = item["bootstrap_95_percent_ci"]
        lines.append(
            f"| {method} | {item['mean_difference']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{item['exact_paired_permutation_p_two_sided']:.3f} |"
        )
    lines.extend(["", "## Protocol Finding", "", "PiEvo's three MBO best-SQ values are identical because the formal run enabled the common generator and evaluated the same 24-candidate pool in each seed. This makes its observed variance artificially small and the current comparison unsuitable for a final significance claim."])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "rerun_required": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
