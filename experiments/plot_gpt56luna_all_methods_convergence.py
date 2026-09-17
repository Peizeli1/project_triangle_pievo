#!/usr/bin/env python3
"""Build a traceable all-method convergence figure from the frozen Luna matrix.

The reducer reads retained JSONL trajectories only.  It does not call the
model, rerun experiments, or mix the formal medium-reasoning track with the
separate exploratory tracks.
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
BASELINE_RUN_MAP = ROOT / "experiments/results/gpt56luna_table1_baselines_20260812/baseline_run_map.json"
FORMAL_RUN_MAP = ROOT / "experiments/results/gpt56luna_formal_run_map_20260813.json"
OUTPUT_JSON = ROOT / "experiments/results/gpt56luna_all_methods_convergence_20260816.json"
OUTPUT_CSV = ROOT / "experiments/results/gpt56luna_all_methods_convergence_20260816.csv"
FIGURE_PDF = ROOT / "paper_iclr2027/figures/fig_gpt56luna_all_methods_convergence.pdf"
FIGURE_PNG = ROOT / "paper_iclr2027/figures/fig_gpt56luna_all_methods_convergence.png"

TASKS = ("MBO", "NHO", "SPO", "TMC")
SEEDS = (0, 1, 2)
BASELINE_METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
METHODS = BASELINE_METHODS + ("pievo", "radp")
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
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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


def baseline_values(path: Path) -> list[float]:
    selected = [
        row
        for row in load_jsonl(path)
        if row.get("success") is True
        and row.get("status") == "primary"
        and finite(row.get("output"))
    ]
    selected.sort(key=lambda row: int(row.get("call_index", row.get("iteration", 0))))
    indices = [int(row.get("call_index", row.get("iteration", 0))) for row in selected]
    if len(selected) != 24 or indices != list(range(1, 25)):
        raise ValueError(f"baseline trajectory is incomplete or non-contiguous: {path}")
    return [float(row["output"]) for row in selected]


def formal_values(path: Path) -> list[float]:
    selected = [
        row
        for row in load_jsonl(path)
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and finite(row.get("output"))
    ]
    selected.sort(key=lambda row: int(row.get("call_index", 0)))
    indices = [int(row.get("call_index", 0)) for row in selected]
    if len(selected) != 24 or indices != list(range(1, 25)):
        raise ValueError(f"formal trajectory is incomplete or non-contiguous: {path}")
    return [float(row["output"]) for row in selected]


def main() -> int:
    baseline_map = load_json(BASELINE_RUN_MAP)
    formal_map_document = load_json(FORMAL_RUN_MAP)
    formal_map = formal_map_document.get("run_map")
    if not isinstance(formal_map, dict):
        raise ValueError(f"missing run_map in {FORMAL_RUN_MAP}")

    cells: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for task in TASKS:
            for seed in SEEDS:
                key = f"{method}:{task}:{seed}"
                if method in BASELINE_METHODS:
                    raw_path = baseline_map.get(key)
                    manifest_name = "run_manifest.json"
                    expected_identity = {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "primary_budget": 24,
                        "model": "gpt-5.6-luna",
                        "stage": "table1_formal_baseline",
                    }
                    trajectory_name = "baseline_search.jsonl"
                    value_loader = baseline_values
                else:
                    raw_path = formal_map.get(key)
                    manifest_name = "run_manifest.json"
                    expected_identity = {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "primary_budget": 24,
                        "model": "gpt-5.6-luna",
                        "track": "gpt56luna",
                        "stage": "formal",
                    }
                    trajectory_name = AUDIT_NAMES[task]
                    value_loader = formal_values
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError(f"missing formal run-map entry: {key}")
                run_dir = Path(raw_path)
                manifest_path = run_dir / manifest_name
                manifest = load_json(manifest_path)
                identity = manifest.get("run_identity", {})
                for name, expected in expected_identity.items():
                    if identity.get(name) != expected:
                        raise ValueError(f"identity mismatch for {key}: {name}={identity.get(name)!r}")
                trajectory_path = run_dir / trajectory_name
                values = value_loader(trajectory_path)
                cells[key] = {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "run_manifest_sha256": sha256(manifest_path),
                    "trajectory": str(trajectory_path.resolve()),
                    "trajectory_sha256": sha256(trajectory_path),
                    "values": values,
                    "running_best_sq": running_normalized(values, task),
                }

    aggregates: dict[str, Any] = {}
    for task in TASKS:
        for method in METHODS:
            selected = [cells[f"{method}:{task}:{seed}"]["running_best_sq"] for seed in SEEDS]
            points = []
            for budget in range(24):
                points.append({"budget": budget + 1, **mean_std([curve[budget] for curve in selected])})
            aggregates[f"{method}:{task}"] = {
                "task": task,
                "method": method,
                "seeds": list(SEEDS),
                "points": points,
            }

    macro_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        points = []
        for budget in range(24):
            values = [
                cells[f"{method}:{task}:{seed}"]["running_best_sq"][budget]
                for task in TASKS
                for seed in SEEDS
            ]
            points.append({"budget": budget + 1, **mean_std(values)})
        macro_by_method[method] = points

    result = {
        "schema_version": "gpt56luna_all_methods_convergence.v1",
        "paper_eligible": True,
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": 24,
        "metric": "normalized best-so-far SQ (%)",
        "normalization_reference": MU_REF,
        "uncertainty": "mean +/- one standard deviation across the three fixed seeds",
        "aggregation": "Each task panel averages the three fixed seed trajectories for each method.",
        "baseline_run_map": str(BASELINE_RUN_MAP.resolve()),
        "baseline_run_map_sha256": sha256(BASELINE_RUN_MAP),
        "formal_run_map": str(FORMAL_RUN_MAP.resolve()),
        "formal_run_map_sha256": sha256(FORMAL_RUN_MAP),
        "cells": cells,
        "aggregates": aggregates,
        "macro_by_method": macro_by_method,
        "claim_boundary": "Formal matched released-evaluator convergence diagnostic for all eight methods; it compares downstream search progress and does not identify a causal gate effect or source-attribution accuracy.",
        "outputs": {
            "figure_pdf": str(FIGURE_PDF.resolve()),
            "figure_png": str(FIGURE_PNG.resolve()),
            "csv": str(OUTPUT_CSV.resolve()),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
                writer.writerow(["macro_method", "ALL", method, point["budget"], f"{point['mean']:.10f}", f"{point['std']:.10f}", point["n"]])

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
        for method in METHODS:
            points = aggregates[f"{method}:{task}"]["points"]
            x = [point["budget"] for point in points]
            mean = [float(point["mean"]) for point in points]
            spread = [float(point["std"]) for point in points]
            is_focus = method in {"pievo", "radp"}
            axis.plot(
                x,
                mean,
                color=COLORS[method],
                linewidth=1.75 if is_focus else 0.95,
                linestyle="--" if method == "radp" else "-",
                alpha=1.0 if is_focus else 0.82,
                label=LABELS[method],
                zorder=4 if is_focus else 2,
            )
            axis.fill_between(
                x,
                [m - s for m, s in zip(mean, spread)],
                [m + s for m, s in zip(mean, spread)],
                color=COLORS[method],
                alpha=0.11 if is_focus else 0.025,
                linewidth=0,
                zorder=1,
            )
        axis.set_title(task)
        axis.set_xlim(1, 24)
        axis.set_xticks((1, 6, 12, 18, 24))
        axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel("Primary evaluations")
    axes[0].set_ylabel("Best-so-far SQ (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.025))
    figure.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.15)
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    print(json.dumps({
        "output_json": str(OUTPUT_JSON.resolve()),
        "output_csv": str(OUTPUT_CSV.resolve()),
        "figure_pdf": str(FIGURE_PDF.resolve()),
        "figure_png": str(FIGURE_PNG.resolve()),
        "validated_cells": len(cells),
        "methods": list(METHODS),
        "status": "complete",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
