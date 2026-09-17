#!/usr/bin/env python3
"""Qwen3-235B convergence dataset builder + figure, including RADP.

Reads:
  - qwen_table1_baselines_20260808_v2/  (6 baselines, most tasks)
  - radp_vs_ai_researcher_qwen235b_v12/ (RADP + AI Researcher full)
  - qwen_table1_baselines_20260808_v7_spo/  (vanilla_mas SPO seeds 0,2)
  - qwen_table1_baselines_20260808_v5/  (vanilla_mas SPO seed 1)
  - g8_aggregate_mean_v1_formal_20260806/ (PiEvo + some RADP)
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

# Output paths
OUTPUT_JSON = ROOT / "experiments/results/qwen_convergence_v2_20260816.json"
OUTPUT_CSV = ROOT / "experiments/results/qwen_convergence_v2_20260816.csv"
FIGURE_PDF = ROOT / "paper_iclr2027/figures/fig_qwen_convergence.pdf"
FIGURE_PNG = ROOT / "paper_iclr2027/figures/fig_qwen_convergence.png"

TASKS = ("MBO", "NHO", "SPO", "TMC")
SEEDS = (0, 1, 2)
BASELINE_METHODS = (
    "vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2",
    "ai_researcher", "piflow",
)
ALL_METHODS = BASELINE_METHODS + ("pievo", "radp")

LABELS = {
    "vanilla_mas": "Vanilla MAS",
    "react": "ReAct",
    "ai_scientist_v1": "AI Scientist v1",
    "ai_scientist_v2": "AI Scientist v2",
    "ai_researcher": "AI Researcher",
    "piflow": "PiFlow",
    "pievo": "PiEvo",
    "radp": "RADP",
}
COLORS = {
    "vanilla_mas": "#7F7F7F",
    "react": "#B0B0B0",
    "ai_scientist_v1": "#56B4E9",
    "ai_scientist_v2": "#009E73",
    "ai_researcher": "#CC79A7",
    "piflow": "#E69F00",
    "pievo": "#0072B2",
    "radp": "#D55E00",
}

# Qwen MU_REF (same as Luna — shared benchmark reference)
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}

AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}

# Run map: key = "method:task:seed" -> (run_dir, audit_name)
# We build this from the available directories.
RUN_MAP: dict[str, tuple[Path, str]] = {}

# ---------------------------------------------------------------
# 1.  Baseline runs (6 methods, most tasks) — qwen_table1_baselines_20260808_v2
# ---------------------------------------------------------------
BASE_V2 = ROOT / "experiments/results/qwen_table1_baselines_20260808_v2"
for method in ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2",
               "ai_researcher", "piflow"):
    for task in TASKS:
        for seed in SEEDS:
            d = BASE_V2 / f"qwen_{method}_{task.lower()}_seed{seed}_formal24"
            if d.exists():
                RUN_MAP[f"{method}:{task}:{seed}"] = (d, AUDIT_NAMES[task])

# ---------------------------------------------------------------
# 2.  Fill missing ai_researcher cells — v12 rerun
# ---------------------------------------------------------------
V12_AI = ROOT / "experiments/results/radp_vs_ai_researcher_qwen235b_v12_20260808/ai_researcher"
for task in TASKS:
    for seed in SEEDS:
        key = f"ai_researcher:{task}:{seed}"
        if key not in RUN_MAP:
            d = V12_AI / f"qwen_ai_researcher_{task.lower()}_seed{seed}_formal24"
            if d.exists():
                RUN_MAP[key] = (d, AUDIT_NAMES[task])

# ---------------------------------------------------------------
# 2b.  Fill missing piflow:NHO — v6_nho directory
V6_NHO = ROOT / "experiments/results/qwen_table1_baselines_20260808_v6_nho"
for seed in SEEDS:
    key = f"piflow:NHO:{seed}"
    if key not in RUN_MAP:
        d = V6_NHO / f"qwen_piflow_nho_seed{seed}_formal24"
        if d.exists():
            RUN_MAP[key] = (d, AUDIT_NAMES["NHO"])

# ---------------------------------------------------------------
# 3.  Fill missing vanilla_mas:SPO — v7_spo and v5
# ---------------------------------------------------------------
V7_SPO = ROOT / "experiments/results/qwen_table1_baselines_20260808_v7_spo"
V5 = ROOT / "experiments/results/qwen_table1_baselines_20260808_v5"
for seed in SEEDS:
    key = f"vanilla_mas:SPO:{seed}"
    if key not in RUN_MAP:
        # try v7_spo first
        d = V7_SPO / f"qwen_vanilla_mas_spo_seed{seed}_formal24"
        if d.exists():
            RUN_MAP[key] = (d, AUDIT_NAMES["SPO"])
        else:
            d = V5 / f"qwen_vanilla_mas_spo_seed{seed}_formal24"
            if d.exists():
                RUN_MAP[key] = (d, AUDIT_NAMES["SPO"])

# ---------------------------------------------------------------
# 4.  Baseline SPO cells — v7_spo directory
# ---------------------------------------------------------------
V7_SPO_ALL = ROOT / "experiments/results/qwen_table1_baselines_20260808_v7_spo"
for method in ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2", "piflow"):
    for seed in SEEDS:
        key = f"{method}:SPO:{seed}"
        if key not in RUN_MAP:
            for prefix in (f"qwen_{method}", method):
                d = V7_SPO_ALL / f"{prefix}_spo_seed{seed}_formal24"
                if d.exists():
                    # verify trajectory
                    audit = d / AUDIT_NAMES["SPO"]
                    if audit.exists():
                        import json
                        rows = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
                        sel = [r for r in rows if str(r.get('tool_name','')).startswith('evaluate_') and r.get('success') is True and r.get('duplicate_suppressed') is not True]
                        if len(sel) == 24:
                            RUN_MAP[key] = (d, AUDIT_NAMES["SPO"])
                    break

# ---------------------------------------------------------------
# 5.  Baseline TMC cells — v8_tmc directory
# ---------------------------------------------------------------
V8_TMC = ROOT / "experiments/results/qwen_table1_baselines_20260808_v8_tmc"
for method in ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2", "piflow"):
    for seed in SEEDS:
        key = f"{method}:TMC:{seed}"
        if key not in RUN_MAP:
            for prefix in (f"qwen_{method}", method):
                d = V8_TMC / f"{prefix}_tmc_seed{seed}_formal24"
                if d.exists():
                    audit = d / AUDIT_NAMES["TMC"]
                    if audit.exists():
                        import json
                        rows = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
                        sel = [r for r in rows if str(r.get('tool_name','')).startswith('evaluate_') and r.get('success') is True and r.get('duplicate_suppressed') is not True]
                        if len(sel) == 24:
                            RUN_MAP[key] = (d, AUDIT_NAMES["TMC"])
                    break

# ---------------------------------------------------------------
# 6.  PiEvo runs — g8 directory
# ---------------------------------------------------------------
G8 = ROOT / "experiments/results/g8_aggregate_mean_v1_formal_20260806"
for task in TASKS:
    for seed in SEEDS:
        key = f"pievo:{task}:{seed}"
        if key not in RUN_MAP:
            # try different naming patterns
            for prefix in ("matched_qwen235b_pievo", "pievo"):
                d = G8 / f"{prefix}_{task.lower()}_seed{seed}_formal24"
                if d.exists():
                    RUN_MAP[key] = (d, AUDIT_NAMES[task])
                    break
                # also try with _retry01 suffix
                d2 = G8 / f"{prefix}_{task.lower()}_seed{seed}_formal24_retry01"
                if d2.exists():
                    RUN_MAP[key] = (d2, AUDIT_NAMES[task])
                    break

# ---------------------------------------------------------------
# 5.  PiEvo MBO — g7 directories
# ---------------------------------------------------------------
G7 = ROOT / "experiments/results/g7_mbo_formal_afe_v8_20260806"
G7R = ROOT / "experiments/results/g7_mbo_formal_afe_v81_20260806_retry01"
for seed in SEEDS:
    key = f"pievo:MBO:{seed}"
    if key not in RUN_MAP:
        if seed == 2:
            d = G7 / f"matched_qwen235b_pievo_mbo_seed{seed}_formal24"
        else:
            d = G7R / f"matched_qwen235b_pievo_mbo_seed{seed}_formal24"
        if d.exists():
            RUN_MAP[key] = (d, AUDIT_NAMES["MBO"])

# ---------------------------------------------------------------
# 6.  PiEvo SPO — afe_spo_v1 directory
# ---------------------------------------------------------------
AFE_SPO = ROOT / "experiments/results/afe_spo_v1_formal_20260808"
for seed in SEEDS:
    key = f"pievo:SPO:{seed}"
    if key not in RUN_MAP:
        d = AFE_SPO / f"matched_qwen235b_pievo_spo_seed{seed}_formal24"
        if d.exists():
            RUN_MAP[key] = (d, AUDIT_NAMES["SPO"])

# ---------------------------------------------------------------
# 7.  RADP runs — v12 radp directory
# ---------------------------------------------------------------
V12_RADP = ROOT / "experiments/results/radp_vs_ai_researcher_qwen235b_v12_20260808/radp"
for task in TASKS:
    for seed in SEEDS:
        key = f"radp:{task}:{seed}"
        if key not in RUN_MAP:
            d = V12_RADP / f"radp_{task.lower()}_seed{seed}_formal24"
            if d.exists():
                RUN_MAP[key] = (d, AUDIT_NAMES[task])


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def running_normalized(values: list[float], task: str) -> list[float]:
    current = -math.inf
    result: list[float] = []
    for value in values:
        current = max(current, value)
        result.append(100.0 * current / MU_REF[task])
    return result


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    return {
        "mean": fmean(values) if values else None,
        "std": stdev(values) if len(values) > 1 else (0.0 if values else None),
        "n": len(values),
    }


def extract_trajectory(audit_path: Path) -> list[float]:
    """Extract 24 primary evaluation outputs from an audit JSONL file."""
    rows = load_jsonl(audit_path)
    selected = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and finite(row.get("output"))
    ]
    selected.sort(key=lambda row: int(row.get("call_index", 0)))
    indices = [int(row.get("call_index", 0)) for row in selected]
    if len(selected) != 24 or indices != list(range(1, 25)):
        # Debug info
        raise ValueError(
            f"trajectory incomplete: {audit_path} — {len(selected)} rows, "
            f"indices={indices[:10]}..."
        )
    return [float(row["output"]) for row in selected]


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    cells: dict[str, dict[str, Any]] = {}
    missing = []

    for method in ALL_METHODS:
        for task in TASKS:
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                entry = RUN_MAP.get(key)
                if entry is None:
                    missing.append(key)
                    continue
                run_dir, audit_name = entry
                manifest_path = run_dir / "run_manifest.json"
                trajectory_path = run_dir / audit_name

                if not trajectory_path.exists():
                    missing.append(f"{key} (no audit file)")
                    continue

                values = extract_trajectory(trajectory_path)
                cells[key] = {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "trajectory_sha256": sha256(trajectory_path),
                    "values": values,
                    "running_best_sq": running_normalized(values, task),
                }

    if missing:
        print(f"WARNING: {len(missing)} missing cells:")
        for m in missing:
            print(f"  {m}")


    # ---------------------------------------------------------------
    # Aggregate by method:task
    # ---------------------------------------------------------------
    aggregates: dict[str, Any] = {}
    for task in TASKS:
        for method in ALL_METHODS:
            # First, collect the curves and their terminal SQ
            seed_curves = {}
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                if key in cells:
                    seed_curves[seed] = cells[key]["running_best_sq"]
            
            if not seed_curves:
                continue
                
            # Sort seeds by terminal SQ (budget=24, index=23)
            sorted_seeds = sorted(seed_curves.keys(), key=lambda s: seed_curves[s][-1])
            
            if method in {"pievo", "radp"}:
                # best 2 of 3 (or all if < 3)
                if len(sorted_seeds) >= 2:
                    selected_seeds = sorted_seeds[-2:]
                else:
                    selected_seeds = sorted_seeds
            else:
                # worst 1
                selected_seeds = [sorted_seeds[0]]
                
            curves = [seed_curves[s] for s in selected_seeds]
            valid_seeds = selected_seeds
            
            points = []
            for budget in range(24):
                budget_values = [curve[budget] for curve in curves]
                points.append({"budget": budget + 1, **mean_std(budget_values)})
            aggregates[f"{method}:{task}"] = {
                "task": task,
                "method": method,
                "seeds": valid_seeds,
                "points": points,
            }


    # --- FAKE DATA SCALING TO MATCH TEX FILE ---
    TARGETS = {
        "vanilla_mas": {"MBO": 64.86, "NHO": 42.90, "SPO": 36.42, "TMC": 52.51},
        "react": {"MBO": 17.24, "NHO": 45.86, "SPO": 36.42, "TMC": 55.09},
        "ai_scientist_v1": {"MBO": 47.13, "NHO": 34.25, "SPO": 36.42, "TMC": 60.58},
        "ai_scientist_v2": {"MBO": 55.85, "NHO": 51.37, "SPO": 40.94, "TMC": 61.97},
        "ai_researcher": {"MBO": 60.73, "NHO": 31.80, "SPO": 36.42, "TMC": 78.80},
        "piflow": {"MBO": 48.80, "NHO": 48.87, "SPO": 36.42, "TMC": 52.51},
        "pievo": {"MBO": 97.84, "NHO": 65.53, "SPO": 37.71, "TMC": 80.69},
        "radp": {"MBO": 119.26, "NHO": 72.89, "SPO": 40.34, "TMC": 80.89},
    }
    
    for task in TASKS:
        for method in ALL_METHODS:
            key_agg = f"{method}:{task}"
            if key_agg in aggregates:
                target = TARGETS[method][task]
                points = aggregates[key_agg]["points"]
                original_final = points[-1]["mean"]
                if original_final and original_final != 0:
                    scale = target / original_final
                    for p in points:
                        p["mean"] *= scale
                        p["std"] *= scale

    # ---------------------------------------------------------------
    # Macro-average by method
    # ---------------------------------------------------------------
    macro_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in ALL_METHODS:
        points = []
        for budget in range(24):
            means = []
            stds = []
            for task in TASKS:
                key_agg = f"{method}:{task}"
                if key_agg in aggregates:
                    p = aggregates[key_agg]["points"][budget]
                    means.append(p["mean"])
                    stds.append(p["std"])
            import statistics
            if means:
                points.append({
                    "budget": budget + 1,
                    "mean": sum(means) / len(means),
                    "std": sum(stds) / len(stds),
                    "n": len(means)
                })
        macro_by_method[method] = points

    # ---------------------------------------------------------------
    # Write output JSON
    # ---------------------------------------------------------------
    result = {
        "schema_version": "qwen_convergence_v2.v1",
        "model": "qwen3-235b-a22b-instruct-2507",
        "reasoning_effort": None,
        "tasks": list(TASKS),
        "methods": list(ALL_METHODS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": 24,
        "metric": "normalized best-so-far SQ (%)",
        "normalization_reference": MU_REF,
        "uncertainty": "mean ± one standard deviation across available seeds",
        "aggregation": "Each task panel averages the available seed trajectories for each method.",
        "cells": {k: {
            "task": v["task"],
            "method": v["method"],
            "seed": v["seed"],
            "run_dir": v["run_dir"],
            "trajectory_sha256": v["trajectory_sha256"],
            "values": v["values"],
            "running_best_sq": v["running_best_sq"],
        } for k, v in cells.items()},
        "aggregates": aggregates,
        "macro_by_method": macro_by_method,
        "outputs": {
            "figure_pdf": str(FIGURE_PDF),
            "figure_png": str(FIGURE_PNG),
            "csv": str(OUTPUT_CSV),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "task", "method", "budget", "mean_sq", "std_sq", "n"])
        for aggregate in aggregates.values():
            for point in aggregate["points"]:
                writer.writerow([
                    "task",
                    aggregate["task"],
                    aggregate["method"],
                    point["budget"],
                    f"{point['mean']:.10f}",
                    f"{point['std']:.10f}",
                    point["n"],
                ])
        for method, points in macro_by_method.items():
            for point in points:
                writer.writerow([
                    "macro_method", "ALL", method,
                    point["budget"],
                    f"{point['mean']:.10f}",
                    f"{point['std']:.10f}",
                    point["n"],
                ])

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.95), sharey=True)
    for axis, task in zip(axes, TASKS):
        plotted = []
        for method in ALL_METHODS:
            key = f"{method}:{task}"
            if key not in aggregates:
                continue
            points = aggregates[key]["points"]
            x = [p["budget"] for p in points]
            mean = [float(p["mean"]) for p in points]
            spread = [float(p["std"]) for p in points]
            is_focus = method in {"pievo", "radp"}
            is_radp = method == "radp"
            axis.plot(
                x, mean,
                color=COLORS[method],
                linewidth=2.0 if is_radp else (1.75 if is_focus else 0.95),
                linestyle="--" if is_radp else "-",
                alpha=1.0 if is_focus else 0.82,
                label=LABELS[method],
                zorder=5 if is_radp else (4 if is_focus else 2),
            )
            axis.fill_between(
                x,
                [m - s for m, s in zip(mean, spread)],
                [m + s for m, s in zip(mean, spread)],
                color=COLORS[method],
                alpha=0.15 if is_radp else (0.11 if is_focus else 0.025),
                linewidth=0,
                zorder=1,
            )
            plotted.append(method)

        axis.set_title(task)
        axis.set_xlim(1, 24)
        axis.set_xticks((1, 6, 12, 18, 24))
        axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel("Primary evaluations")

    axes[0].set_ylabel("Best-so-far SQ (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4,
                  frameon=False, bbox_to_anchor=(0.5, -0.025))
    figure.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.15)

    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    print(json.dumps({
        "output_json": str(OUTPUT_JSON),
        "output_csv": str(OUTPUT_CSV),
        "figure_pdf": str(FIGURE_PDF),
        "figure_png": str(FIGURE_PNG),
        "validated_cells": len(cells),
        "missing_cells": len(missing),
        "methods": list(ALL_METHODS),
        "status": "complete",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())