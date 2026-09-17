#!/usr/bin/env python3
"""Gemini 2.5 Flash Lite convergence figure, same style as Luna all-methods."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = ROOT / "experiments/results/gemini_convergence_20260816.json"
FIGURE_PDF = ROOT / "paper_iclr2027/figures/fig_gemini_convergence.pdf"
FIGURE_PNG = ROOT / "paper_iclr2027/figures/fig_gemini_convergence.png"

TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2",
           "ai_researcher", "piflow", "pievo", "radp")
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


def main() -> int:
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    aggregates = data["aggregates"]
    tasks = data.get("tasks", TASKS)
    methods = data.get("methods", METHODS)

    axis_methods: dict[str, list[str]] = {}
    for t in tasks:
        axis_methods[t] = [m for m in methods if f"{m}:{t}" in aggregates]

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
    for axis, task in zip(axes, tasks):
        for method in axis_methods[task]:
            points = aggregates[f"{method}:{task}"]["points"]
            x = [p["budget"] for p in points]
            mean = [float(p["mean"]) for p in points]
            spread = [float(p["std"]) for p in points]
            is_focus = method in {"pievo", "radp"}
            axis.plot(
                x, mean,
                color=COLORS.get(method, "#333333"),
                linewidth=1.75 if is_focus else 0.95,
                linestyle="--" if method == "radp" else "-",
                alpha=1.0 if is_focus else 0.82,
                label=LABELS.get(method, method),
                zorder=4 if is_focus else 2,
            )
            axis.fill_between(
                x,
                [m - s for m, s in zip(mean, spread)],
                [m + s for m, s in zip(mean, spread)],
                color=COLORS.get(method, "#333333"),
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
    figure.legend(
        handles, labels, loc="lower center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, -0.025),
    )
    figure.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.15)

    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    print(json.dumps({
        "output_pdf": str(FIGURE_PDF),
        "output_png": str(FIGURE_PNG),
        "input_json": str(INPUT_JSON),
        "tasks": list(tasks),
        "methods": {t: axis_methods[t] for t in tasks},
        "status": "complete",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())