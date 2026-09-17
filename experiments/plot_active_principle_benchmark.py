#!/usr/bin/env python3
"""Publication-style plots for the RADP-v1 controlled benchmark.

The script reads the benchmark's auditable CSV/JSON artifacts and exports
vector PDF plus 300-dpi PNG figures.  It intentionally uses a colorblind-safe
palette and keeps the two seeds visible as paired points where appropriate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ["source_blind", "fixed_gated", "active_gated", "active_no_backtrack"]
POLICY_LABELS = {
    "source_blind": "Source-blind",
    "fixed_gated": "Fixed gate",
    "active_gated": "Active + backtrack",
    "active_no_backtrack": "Active, no backtrack",
}
COLORS = {
    "source_blind": "#B0BEC5",
    "fixed_gated": "#0072B2",
    "active_gated": "#D55E00",
    "active_no_backtrack": "#009E73",
}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if str(value).strip().lower() in {"true", "false"}:
        return float(str(value).strip().lower() == "true")
    return float(value)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_false_expansion(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    selected = [row for row in rows if row["budget"] == "48"]
    conditions = ["noise", "surrogate_failure", "local_artifact", "mechanism_reversal"]
    labels = ["Noise", "Surrogate\nfailure", "Local\nartifact", "Reversal"]
    x = np.arange(len(conditions))
    width = 0.8 / len(POLICIES)
    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    for i, policy in enumerate(POLICIES):
        values = []
        for condition in conditions:
            values.append(mean(
                as_float(row["false_expansion"])
                for row in selected
                if row["condition"] == condition and row["policy"] == policy
            ))
        bars = ax.bar(
            x + (i - (len(POLICIES) - 1) / 2) * width,
            values,
            width * 0.9,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, value + 0.04),
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("False principle expansion rate")
    ax.set_title("Source-aware validation prevents unsupported expansion")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    save(fig, output_dir, "fig_false_expansion_48")


def plot_regret_scaling(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    selected = [row for row in rows if row["condition"] in {"surrogate_failure", "mechanism_reversal"}]
    budgets = ["8", "24", "48"]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65), sharey=False)
    for ax, condition, title in zip(
        axes,
        ["surrogate_failure", "mechanism_reversal"],
        ["Surrogate failure", "Mechanism reversal"],
    ):
        for policy in POLICIES:
            means = []
            for budget in budgets:
                means.append(mean(
                    as_float(row["cumulative_regret"])
                    for row in selected
                    if row["condition"] == condition
                    and row["policy"] == policy
                    and row["budget"] == budget
                ))
            ax.plot(
                [int(b) for b in budgets],
                means,
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=POLICY_LABELS[policy],
                color=COLORS[policy],
            )
        ax.set_title(title)
        ax.set_xlabel("Primary evaluations")
        ax.set_xticks([8, 24, 48])
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Cumulative true regret")
    axes[1].legend(ncol=2, loc="upper center", bbox_to_anchor=(-0.05, -0.25), frameon=False)
    fig.suptitle("Budget scaling under source ambiguity", y=1.03)
    save(fig, output_dir, "fig_budget_regret")


def plot_state_accuracy(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    selected = [row for row in rows if row["condition"] != "clean"]
    budgets = ["8", "24", "48"]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for policy in POLICIES:
        values = []
        for budget in budgets:
            values.append(mean(
                as_float(row["final_state_correct"])
                for row in selected
                if row["policy"] == policy and row["budget"] == budget
            ))
        ax.plot(
            [int(b) for b in budgets],
            values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_xlabel("Primary evaluations")
    ax.set_ylabel("Final state accuracy")
    ax.set_xticks([8, 24, 48])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.18)
    ax.set_title("Correct KEEP / EXPAND / BACKTRACK state")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    save(fig, output_dir, "fig_state_accuracy_scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.benchmark_dir / "trial_results.csv")
    checkpoint = read_csv(args.benchmark_dir / "checkpoint_metrics.csv")
    (args.output_dir / "plot_input_manifest.json").write_text(
        json.dumps({
            "benchmark_dir": str(args.benchmark_dir.resolve()),
            "trial_rows": len(rows),
            "checkpoint_rows": len(checkpoint),
            "figures": [
                "fig_false_expansion_48",
                "fig_budget_regret",
                "fig_state_accuracy_scaling",
            ],
        }, indent=2),
        encoding="utf-8",
    )
    plot_false_expansion(rows, args.output_dir)
    plot_regret_scaling(rows, args.output_dir)
    plot_state_accuracy(rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "figures": 3}, indent=2))


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    })
    main()
