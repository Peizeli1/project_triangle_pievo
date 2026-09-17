#!/usr/bin/env python3
"""Plot convergence figure for any model track: 5 panels (4 tasks + macro), per-task y-axis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_convergence(input_json: Path, figure_pdf: Path, figure_png: Path,
                     model_label: str = "") -> dict:
    """Generate a 5-panel convergence figure with per-task y-axis."""
    data = json.loads(input_json.read_text(encoding="utf-8"))
    aggregates = data.get("aggregates", {})
    macro = data.get("macro_by_method", {})
    all_tasks = ["NHO", "MBO", "SPO", "TMC"]
    all_methods = data.get("methods", [])

    LABELS = {
        "vanilla_mas": "Vanilla",
        "react": "ReAct",
        "ai_scientist_v1": "The-AI-Scientist-v1",
        "ai_scientist_v2": "The-AI-Scientist-v2",
        "ai_researcher": "AI-Researcher",
        "piflow": "PiFlow",
        "pievo": "PiEvo",
        "radp": "RADP"
    }
    COLORS = {
        "vanilla_mas": "black",
        "react": "orange",
        "ai_scientist_v1": "gray",
        "ai_scientist_v2": "lightgray",
        "ai_researcher": "darkblue",
        "piflow": "purple",
        "pievo": "darkred",
        "radp": "red"
    }
    MARKERS = {
        "vanilla_mas": "o",
        "react": "x",
        "ai_scientist_v1": "s",
        "ai_scientist_v2": "+",
        "ai_researcher": "D",
        "piflow": "*",
        "pievo": "^",
        "radp": "v"
    }

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                          "axes.titlesize": 11, "axes.labelsize": 10,
                          "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=False)

    for ax, task in zip(axes, all_tasks):
        items = [(k.split(":")[0], v) for k, v in aggregates.items() if f":{task}" in k]

        for method, val in items:
            if isinstance(val, dict):
                pts = val.get("points", [])
            elif isinstance(val, list):
                pts = val
            else:
                continue
            if not pts:
                continue
            x = [p["budget"] for p in pts]
            mean = [float(p["mean"]) for p in pts]
            spread = [float(p["std"]) for p in pts]
            is_focus = method in ("pievo", "radp")

            ax.plot(x, mean, color=COLORS.get(method, "#333"),
                    marker=MARKERS.get(method, "o"), markersize=4.5,
                    linewidth=2.0 if is_focus else 1.5,
                    linestyle="--",
                    alpha=1.0 if is_focus else 0.8,
                    label=LABELS.get(method, method),
                    zorder=5 if is_focus else 2)
            ax.fill_between(x, [m - s for m, s in zip(mean, spread)],
                            [m + s for m, s in zip(mean, spread)],
                            color=COLORS.get(method, "#333"),
                            alpha=0.15 if is_focus else 0.05,
                            linewidth=0, zorder=1)

        ax.set_title(f"Trajectory on {task}", fontweight="bold")
        ax.set_xlim(1, 24)
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, color="#E5E5E5", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
        ax.set_xlabel("Iteration", fontweight="bold", fontsize=11)
        ax.set_ylabel("Cumulative SQ Max (%)", fontweight="bold", fontsize=11)

        if task in ["MBO", "SPO", "TMC"]:
            ax.legend(title="Method", loc="lower right", frameon=True,
                      facecolor="white", edgecolor="#CCCCCC", fontsize=7, title_fontsize=8)

    fig.tight_layout(w_pad=1.5)

    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figure_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return {
        "figure_pdf": str(figure_pdf),
        "figure_png": str(figure_png),
        "input_json": str(input_json),
        "tasks": list(all_tasks) + ["Macro Avg"],
        "methods": list(all_methods),
    }


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    configs = {
        "luna": {
            "input": ROOT / "experiments/results/gpt56luna_all_methods_convergence_20260816.json",
            "pdf": ROOT / "paper_iclr2027/figures/fig_gpt56luna_convergence.pdf",
            "png": ROOT / "paper_iclr2027/figures/fig_gpt56luna_convergence.png",
        },
        "qwen": {
            "input": ROOT / "experiments/results/qwen_convergence_v2_20260816.json",
            "pdf": ROOT / "paper_iclr2027/figures/fig_qwen_convergence.pdf",
            "png": ROOT / "paper_iclr2027/figures/fig_qwen_convergence.png",
        },
        "gemini": {
            "input": ROOT / "experiments/results/gemini_convergence_v2_20260816.json",
            "pdf": ROOT / "paper_iclr2027/figures/fig_gemini_convergence.pdf",
            "png": ROOT / "paper_iclr2027/figures/fig_gemini_convergence.png",
        },
    }

    models = sys.argv[1:] if len(sys.argv) > 1 else list(configs)
    for model in models:
        if model in configs:
            cfg = configs[model]
            result = plot_convergence(cfg["input"], cfg["pdf"], cfg["png"], model_label=model)
            print(f"✅ {model}: {result['figure_pdf']}")
        else:
            print(f"Unknown model: {model}")