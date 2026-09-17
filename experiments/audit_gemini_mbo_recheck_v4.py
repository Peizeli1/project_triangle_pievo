#!/usr/bin/env python3
"""Audit a fresh Gemini MBO AD-constrained matrix and test RADP differences."""

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


SEEDS = tuple(range(8))
MU_REF = 6.5
PRIMARY_BUDGET = 24
AD_CONTRACT = "mbo_structural_envelope_v2"
METHODS = ("pievo", "radp", "piflow")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def primary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in rows
            if str(row.get("tool_name", "")).startswith("evaluate_")
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and finite(row.get("output"))
        ],
        key=lambda row: int(row.get("call_index", 0)),
    )


def trajectory(values: list[float]) -> tuple[float, float]:
    running: list[float] = []
    current = -math.inf
    for value in values:
        current = max(current, value)
        running.append(current)
    return max(values) / MU_REF * 100.0, statistics.fmean(running) / MU_REF * 100.0


def read_matched_cell(root: Path, method: str, seed: int) -> dict[str, Any]:
    run_dir = root / f"gemini25flash_nothinking_{method}_mbo_seed{seed}_formal24"
    manifest = load_json(run_dir / "run_manifest.json")
    rows = load_jsonl(run_dir / "official_chembl_tool_audit.jsonl")
    primary = primary_rows(rows)
    ad_rejected = [row for row in rows if row.get("record_type") == "applicability_domain_rejection"]
    identity = manifest.get("run_identity", {})
    runtime = manifest.get("runtime", {})
    complete = bool(
        manifest.get("status") == "completed"
        and identity.get("task") == "MBO"
        and identity.get("method") == method
        and identity.get("seed") == seed
        and identity.get("primary_budget") == PRIMARY_BUDGET
        and identity.get("applicability_domain_contract") == AD_CONTRACT
        and runtime.get("all_bridge_calls_no_thinking") is True
        and runtime.get("all_bridge_calls_forwarded_seed") is True
        and len(primary) == PRIMARY_BUDGET
        and len({row.get("canonical_smiles") for row in primary}) == PRIMARY_BUDGET
    )
    if not complete:
        raise ValueError(f"incomplete or invalid matched cell: {run_dir}")
    sq, auoc = trajectory([float(row["output"]) for row in primary])
    return {
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "sq_percent": sq,
        "auoc_percent": auoc,
        "primary": len(primary),
        "ad_rejected": len(ad_rejected),
        "duplicate": sum(row.get("duplicate_suppressed") is True for row in rows),
        "evaluator_actions": len([row for row in rows if str(row.get("tool_name", "")).startswith("evaluate_")]),
    }


def read_baseline_cell(root: Path, seed: int) -> dict[str, Any]:
    run_dir = root / f"gemini25flash_nothinking_piflow_mbo_seed{seed}_formal24"
    manifest = load_json(run_dir / "run_manifest.json")
    rows = load_jsonl(run_dir / "official_chembl_tool_audit.jsonl")
    primary = primary_rows(rows)
    ad_rejected = [row for row in rows if row.get("record_type") == "applicability_domain_rejection"]
    identity = manifest.get("run_identity", {})
    runtime = manifest.get("runtime", {})
    complete = bool(
        manifest.get("status") == "completed"
        and identity.get("task") == "MBO"
        and identity.get("method") == "piflow"
        and identity.get("seed") == seed
        and identity.get("primary_budget") == PRIMARY_BUDGET
        and runtime.get("applicability_domain_contract") == AD_CONTRACT
        and len(primary) == PRIMARY_BUDGET
        and len({row.get("canonical_smiles") for row in primary}) == PRIMARY_BUDGET
    )
    if not complete:
        raise ValueError(f"incomplete or invalid baseline cell: {run_dir}")
    sq, auoc = trajectory([float(row["output"]) for row in primary])
    return {
        "method": "piflow",
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "sq_percent": sq,
        "auoc_percent": auoc,
        "primary": len(primary),
        "ad_rejected": len(ad_rejected),
        "duplicate": sum(row.get("duplicate_suppressed") is True for row in rows),
        "evaluator_actions": len([row for row in rows if str(row.get("tool_name", "")).startswith("evaluate_")]),
    }


def exact_independent_permutation(a: np.ndarray, b: np.ndarray, alternative: str) -> float:
    combined = np.concatenate([a, b])
    n = len(a)
    observed = float(np.mean(a) - np.mean(b))
    values: list[float] = []
    for selected in itertools.combinations(range(len(combined)), n):
        mask = np.zeros(len(combined), dtype=bool)
        mask[list(selected)] = True
        values.append(float(np.mean(combined[mask]) - np.mean(combined[~mask])))
    if alternative == "greater":
        return float(np.mean(np.array(values) >= observed - 1e-12))
    if alternative == "less":
        return float(np.mean(np.array(values) <= observed + 1e-12))
    return float(np.mean(np.abs(values) >= abs(observed) - 1e-12))


def exact_paired_sign_permutation(differences: np.ndarray, alternative: str) -> float:
    observed = float(np.mean(differences))
    values = np.array(
        [
            float(np.mean(differences * np.array(signs)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    if alternative == "greater":
        return float(np.mean(values >= observed - 1e-12))
    if alternative == "less":
        return float(np.mean(values <= observed + 1e-12))
    return float(np.mean(np.abs(values) >= abs(observed) - 1e-12))


def bootstrap_paired_difference_ci(
    differences: np.ndarray, *, seed: int = 20260809, samples: int = 100000
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compare(radp: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    delta = radp - reference
    welch = stats.ttest_ind(radp, reference, equal_var=False)
    paired = stats.ttest_rel(radp, reference)
    try:
        wilcoxon = stats.wilcoxon(delta, alternative="two-sided", method="exact")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = None
    return {
        "n_radp": len(radp),
        "n_reference": len(reference),
        "mean_delta_radp_minus_reference": float(np.mean(delta)),
        "std_delta": float(np.std(delta, ddof=1)),
        "bootstrap_95_percent_ci_paired": bootstrap_paired_difference_ci(delta),
        "exact_independent_permutation_p_two_sided": exact_independent_permutation(radp, reference, "two-sided"),
        "exact_independent_permutation_p_radp_greater": exact_independent_permutation(radp, reference, "greater"),
        "exact_paired_sign_permutation_p_two_sided": exact_paired_sign_permutation(delta, "two-sided"),
        "exact_paired_sign_permutation_p_radp_greater": exact_paired_sign_permutation(delta, "greater"),
        "wilcoxon_exact_p_two_sided": wilcoxon_p,
        "welch_t_p_two_sided_descriptive": float(welch.pvalue),
        "paired_t_p_two_sided_descriptive": float(paired.pvalue),
        "all_seed_differences_positive": bool(np.all(delta > 0)),
    }


def stats_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def holm_adjust_paired(comparisons: dict[str, dict[str, Any]]) -> None:
    """Apply Holm correction within one predeclared metric family."""
    ordered = sorted(
        (item["exact_paired_sign_permutation_p_two_sided"], key)
        for key, item in comparisons.items()
    )
    adjusted: dict[str, float] = {}
    for index, (p_value, key) in enumerate(ordered):
        adjusted[key] = float(min(1.0, p_value * (len(ordered) - index)))
    for key, item in comparisons.items():
        item["holm_adjusted_exact_paired_sign_permutation_p_two_sided"] = adjusted[key]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    cells = [read_matched_cell(args.matched_root.resolve(), method, seed) for method in ("pievo", "radp") for seed in SEEDS]
    cells.extend(read_baseline_cell(args.baseline_root.resolve(), seed) for seed in SEEDS)
    by_method = {method: [row for row in cells if row["method"] == method] for method in METHODS}
    values = {
        method: np.array([row["sq_percent"] for row in by_method[method]], dtype=float)
        for method in METHODS
    }
    auoc_values = {
        method: np.array([row["auoc_percent"] for row in by_method[method]], dtype=float)
        for method in METHODS
    }
    comparisons = {
        f"radp_vs_{reference}": compare(values["radp"], values[reference])
        for reference in ("pievo", "piflow")
    }
    auoc_comparisons = {
        f"radp_vs_{reference}": compare(auoc_values["radp"], auoc_values[reference])
        for reference in ("pievo", "piflow")
    }
    # SQ is the primary family; AUOC is reported as a separately corrected
    # secondary family rather than pooling metrics into one post-hoc score.
    holm_adjust_paired(comparisons)
    holm_adjust_paired(auoc_comparisons)
    for key, item in comparisons.items():
        ci = item["bootstrap_95_percent_ci_paired"]
        item["radp_superiority_supported"] = bool(
            item["holm_adjusted_exact_paired_sign_permutation_p_two_sided"] < 0.05 and ci[0] > 0
        )
    for item in auoc_comparisons.values():
        ci = item["bootstrap_95_percent_ci_paired"]
        item["radp_superiority_supported_secondary"] = bool(
            item["holm_adjusted_exact_paired_sign_permutation_p_two_sided"] < 0.05
            and ci[0] > 0
        )
    output = {
        "schema_version": "gemini_mbo_recheck_v4_statistics.v2",
        "model": "gemini-2.5-flash-no-thinking",
        "task": "MBO",
        "seeds": list(SEEDS),
        "primary_budget": PRIMARY_BUDGET,
        "applicability_domain_contract": AD_CONTRACT,
        "cells": cells,
        "method_summary_sq_percent": {method: stats_summary(values[method].tolist()) for method in METHODS},
        "method_summary_auoc_percent": {
            method: stats_summary(auoc_values[method].tolist()) for method in METHODS
        },
        "operational_summary": {
            method: {
                "ad_rejections": stats_summary([float(row["ad_rejected"]) for row in by_method[method]]),
                "duplicates": stats_summary([float(row["duplicate"]) for row in by_method[method]]),
                "evaluator_actions": stats_summary([float(row["evaluator_actions"]) for row in by_method[method]]),
            }
            for method in METHODS
        },
        "comparisons_sq_primary": comparisons,
        "comparisons_auoc_secondary": auoc_comparisons,
        "decision": {
            "current_strongest_mean_sq": max((values[method].mean(), method) for method in METHODS)[1],
            "radp_is_sq_mean_leader": bool(values["radp"].mean() > max(values["pievo"].mean(), values["piflow"].mean())),
            "final_sota_claim_supported": bool(all(item["radp_superiority_supported"] for item in comparisons.values())),
            "claim_boundary": "Fresh AD-constrained matched-evaluator MBO recheck only; not exact hidden paper leaderboard recovery.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Gemini MBO Statistical Audit",
        "",
        "All cells are fresh AD-constrained runs with 8 seeds and 24 primary evaluations per cell.",
        "",
        "## SQ Mean +/- SD",
        "",
        "| Method | Seed values | Mean | Std |",
        "|---|---|---:|---:|",
    ]
    for method in METHODS:
        summary = output["method_summary_sq_percent"][method]
        seed_values = ", ".join(f"{value:.2f}" for value in values[method])
        lines.append(f"| {method} | {seed_values} | {summary['mean']:.2f} | {summary['std']:.2f} |")
    lines.extend(["", "## SQ Pairwise Tests", "", "The paired sign-permutation test is primary because each method is aligned by seed. Independent permutation and Welch t are sensitivity analyses.", "", "| Comparison | Mean delta | Paired bootstrap 95% CI | Paired exact p | Holm p |", "|---|---:|---:|---:|---:|"])
    for key, item in comparisons.items():
        ci = item["bootstrap_95_percent_ci_paired"]
        lines.append(f"| {key} | {item['mean_delta_radp_minus_reference']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {item['exact_paired_sign_permutation_p_two_sided']:.4f} | {item['holm_adjusted_exact_paired_sign_permutation_p_two_sided']:.4f} |")
    lines.extend(["", "## AUOC Pairwise Tests", "", "AUOC is a secondary metric with Holm correction across the same two predeclared comparisons.", "", "| Comparison | Mean delta | Paired bootstrap 95% CI | Paired exact p | Holm p |", "|---|---:|---:|---:|---:|"])
    for key, item in auoc_comparisons.items():
        ci = item["bootstrap_95_percent_ci_paired"]
        lines.append(f"| {key} | {item['mean_delta_radp_minus_reference']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {item['exact_paired_sign_permutation_p_two_sided']:.4f} | {item['holm_adjusted_exact_paired_sign_permutation_p_two_sided']:.4f} |")
    lines.extend(["", "## Operational Audit", "", "| Method | Mean AD rejections | Mean duplicates | Mean evaluator actions |", "|---|---:|---:|---:|"])
    for method in METHODS:
        summary = output["operational_summary"][method]
        lines.append(f"| {method} | {summary['ad_rejections']['mean']:.2f} | {summary['duplicates']['mean']:.2f} | {summary['evaluator_actions']['mean']:.2f} |")
    lines.extend(["", "## Decision", "", f"Current highest mean SQ: `{output['decision']['current_strongest_mean_sq']}`.", f"RADP SQ mean leader: `{output['decision']['radp_is_sq_mean_leader']}`.", f"RADP superiority supported against both predeclared references: `{output['decision']['final_sota_claim_supported']}`.", "", "The primary decision requires a positive paired bootstrap lower bound and Holm-adjusted two-sided paired sign-permutation p < 0.05. No seed may be selected or discarded after observing outcomes."])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "decision": output["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
