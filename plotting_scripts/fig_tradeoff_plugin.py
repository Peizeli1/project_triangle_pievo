#!/usr/bin/env python3
"""Generate publication-ready PDF figure and 1-page PDF summary report for advisor review.

Based on Table 7 (Triage as a Plug-in across Autonomous Research Harnesses)
- X-axis: Token Cost (k tokens) and Execution Runtime (seconds)
- Y-axis: Task Score (raw) and Verified Score (grounded)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# -------------------------------------------------------------
# Data Definition (Table 7 in iclr2027_conference.tex)
# -------------------------------------------------------------
harnesses = ["ResearchHarness", "AI-Researcher", "Claude Code"]
variants = ["Base", "+ Self-Review", "+ Triage (Ours)"]

data = {
    "ResearchHarness": {
        "tokens": [14.2, 18.6, 21.4],
        "runtime": [48.5, 62.1, 71.3],
        "task_score": [0.286, 0.301, 0.345],
        "verified_score": [0.230, 0.248, 0.328],
        "false_adm": [38.4, 35.2, 14.2],
        "rev_acc": [44.2, 48.5, 68.4],
    },
    "AI-Researcher": {
        "tokens": [16.5, 22.1, 25.8],
        "runtime": [56.2, 74.8, 86.5],
        "task_score": [0.332, 0.348, 0.389],
        "verified_score": [0.284, 0.305, 0.371],
        "false_adm": [32.6, 29.8, 11.8],
        "rev_acc": [51.8, 56.2, 73.6],
    },
    "Claude Code": {
        "tokens": [18.8, 25.4, 29.2],
        "runtime": [64.0, 85.2, 98.4],
        "task_score": [0.348, 0.365, 0.412],
        "verified_score": [0.298, 0.321, 0.395],
        "false_adm": [29.4, 26.8, 9.6],
        "rev_acc": [55.4, 60.2, 78.2],
    },
}

colors = {
    "ResearchHarness": "#D95F02",  # Rust orange
    "AI-Researcher": "#1F78B4",    # Classic blue
    "Claude Code": "#2CA02C",      # Forest green
}

markers = {
    "Base": "o",
    "+ Self-Review": "s",
    "+ Triage (Ours)": "*",
}

marker_sizes = {
    "Base": 65,
    "+ Self-Review": 75,
    "+ Triage (Ours)": 180,
}


def draw_tradeoff_figure(fig, ax1, ax2):
    """Draw the 2-panel trade-off curves."""
    ax1.set_title("(a) Token Cost vs. Task & Verified Scores", fontsize=11, fontweight="bold", pad=8)
    ax2.set_title("(b) Runtime vs. Task & Verified Scores", fontsize=11, fontweight="bold", pad=8)

    for h in harnesses:
        col = colors[h]
        t = data[h]["tokens"]
        rt = data[h]["runtime"]
        ts = data[h]["task_score"]
        vs = data[h]["verified_score"]

        # --- Panel (a): Token Cost ---
        # Task Score (raw) - dotted line
        ax1.plot(t, ts, color=col, linestyle=":", linewidth=1.5, alpha=0.6)
        # Verified Score (verified) - solid line
        ax1.plot(t, vs, color=col, linestyle="-", linewidth=2.2, alpha=0.9)

        # Markers
        for i, var in enumerate(variants):
            # Task score marker (lighter)
            ax1.scatter(t[i], ts[i], color=col, marker=markers[var], s=marker_sizes[var]*0.5,
                        edgecolor="gray", alpha=0.5, zorder=3)
            # Verified score marker (bold)
            ax1.scatter(t[i], vs[i], color=col, marker=markers[var], s=marker_sizes[var],
                        edgecolor="#222222", linewidth=1.0, zorder=5)

        # Annotate + Triage points
        ax1.annotate(f"{h}\n(+Triage)", xy=(t[2], vs[2]), xytext=(t[2] + 0.6, vs[2] - 0.008),
                     fontsize=8, color=col, fontweight="bold")

        # --- Panel (b): Runtime ---
        # Task Score
        ax2.plot(rt, ts, color=col, linestyle=":", linewidth=1.5, alpha=0.6)
        # Verified Score
        ax2.plot(rt, vs, color=col, linestyle="-", linewidth=2.2, alpha=0.9)

        # Markers
        for i, var in enumerate(variants):
            ax2.scatter(rt[i], ts[i], color=col, marker=markers[var], s=marker_sizes[var]*0.5,
                        edgecolor="gray", alpha=0.5, zorder=3)
            ax2.scatter(rt[i], vs[i], color=col, marker=markers[var], s=marker_sizes[var],
                        edgecolor="#222222", linewidth=1.0, zorder=5)

        # Annotate + Triage points with clean spacing
        offset_y = -0.012 if h == "Claude Code" else -0.008
        ax2.annotate(f"{h}\n(+Triage)", xy=(rt[2], vs[2]), xytext=(rt[2] + 2.0, vs[2] + offset_y),
                     fontsize=8, color=col, fontweight="bold")

    # Shading the "Verified Gap" reduction for one representative to explain trade-off
    ax1.annotate("Verified Gap collapses\nwith +Triage gate",
                 xy=(29.2, 0.403), xytext=(21.5, 0.425),
                 fontsize=8.5, color="#2CA02C", fontweight="semibold",
                 arrowprops=dict(arrowstyle="->", color="#2CA02C", lw=1.0))

    ax1.set_xlabel("Token Cost (k tokens / task)", fontsize=9.5, fontweight="semibold")
    ax1.set_ylabel("Optimization Score", fontsize=9.5, fontweight="semibold")
    ax1.set_xlim(12, 33)
    ax1.set_ylim(0.20, 0.45)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_xlabel("Execution Runtime (seconds / task)", fontsize=9.5, fontweight="semibold")
    ax2.set_ylabel("Optimization Score", fontsize=9.5, fontweight="semibold")
    ax2.set_xlim(40, 115)
    ax2.set_ylim(0.20, 0.45)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Custom unified legend
    legend_elements = [
        Line2D([0], [0], color="#555555", lw=2.2, linestyle="-", label="Verified Score (Real)"),
        Line2D([0], [0], color="#555555", lw=1.5, linestyle=":", label="Task Score (Reported)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markeredgecolor="k", markersize=7, label="Base"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888", markeredgecolor="k", markersize=7, label="+ Self-Review"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888888", markeredgecolor="k", markersize=11, label="+ Triage (Ours)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")


def generate_standalone_figure_pdf(out_path: Path):
    """Generate the standalone 2-panel figure PDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=300)
    draw_tradeoff_figure(fig, ax1, ax2)
    fig.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    # also save png for easy preview
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[1/2] Standalone Figure PDF saved to: {out_path}")


if __name__ == "__main__":
    generate_standalone_figure_pdf(Path("fig_tradeoff_plugin.pdf"))
