import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from Table 7
harnesses = ["ResearchHarness", "AI-Researcher", "Claude Code"]
variants = ["Base", "+ Self-Review", "+ Triage (Ours)"]

task_score = {
    "ResearchHarness": [0.286, 0.301, 0.345],
    "AI-Researcher": [0.332, 0.348, 0.389],
    "Claude Code": [0.348, 0.365, 0.412]
}

verified_score = {
    "ResearchHarness": [0.230, 0.248, 0.328],
    "AI-Researcher": [0.284, 0.305, 0.371],
    "Claude Code": [0.298, 0.321, 0.395]
}

token_cost = {
    "ResearchHarness": [14.2, 18.6, 21.4],
    "AI-Researcher": [16.5, 22.1, 25.8],
    "Claude Code": [18.8, 25.4, 29.2]
}

runtime = {
    "ResearchHarness": [48.5, 62.1, 71.3],
    "AI-Researcher": [56.2, 74.8, 86.5],
    "Claude Code": [64.0, 85.2, 98.4]
}

# Style configurations
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "grid.alpha": 0.4,
})

colors = {
    "ResearchHarness": "#D55E00", # Vermillion
    "AI-Researcher": "#0072B2",   # Blue
    "Claude Code": "#009E73"      # Green
}

markers = {
    "Base": "o",
    "+ Self-Review": "s",
    "+ Triage (Ours)": "*"
}
sizes = {
    "Base": 50,
    "+ Self-Review": 50,
    "+ Triage (Ours)": 150
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Token Cost vs Scores
for harness in harnesses:
    # Task Score
    ax1.plot(token_cost[harness], task_score[harness], color=colors[harness], 
             linestyle=":", linewidth=1.5, alpha=0.6, label=f"{harness} (Task)")
    # Verified Score
    ax1.plot(token_cost[harness], verified_score[harness], color=colors[harness], 
             linestyle="-", linewidth=2, label=f"{harness} (Verified)")
    
    # Scatter points for variants
    for i, variant in enumerate(variants):
        # Only add label for the first harness to avoid legend duplication of markers
        lbl = variant if harness == harnesses[0] else ""
        ax1.scatter(token_cost[harness][i], verified_score[harness][i], 
                    color=colors[harness], marker=markers[variant], s=sizes[variant], 
                    edgecolor="black", zorder=5, label=lbl if "Task" not in lbl else "")
        ax1.scatter(token_cost[harness][i], task_score[harness][i], 
                    color=colors[harness], marker=markers[variant], s=sizes[variant]*0.6, 
                    edgecolor="black", alpha=0.6, zorder=4)

# Plot 2: Runtime vs Scores
for harness in harnesses:
    # Task Score
    ax2.plot(runtime[harness], task_score[harness], color=colors[harness], 
             linestyle=":", linewidth=1.5, alpha=0.6)
    # Verified Score
    ax2.plot(runtime[harness], verified_score[harness], color=colors[harness], 
             linestyle="-", linewidth=2)
    
    # Scatter points for variants
    for i, variant in enumerate(variants):
        ax2.scatter(runtime[harness][i], verified_score[harness][i], 
                    color=colors[harness], marker=markers[variant], s=sizes[variant], 
                    edgecolor="black", zorder=5)
        ax2.scatter(runtime[harness][i], task_score[harness][i], 
                    color=colors[harness], marker=markers[variant], s=sizes[variant]*0.6, 
                    edgecolor="black", alpha=0.6, zorder=4)

# Formatting Ax1
ax1.set_xlabel("Token Cost (k tokens)")
ax1.set_ylabel("Score")
ax1.set_title("Token Cost vs. Task/Verified Score")
ax1.set_ylim(0.20, 0.45) # Reasonable non-exaggerated range
ax1.grid(True, linestyle="--")

# Formatting Ax2
ax2.set_xlabel("Runtime (seconds)")
ax2.set_ylabel("Score")
ax2.set_title("Runtime vs. Task/Verified Score")
ax2.set_ylim(0.20, 0.45) # Reasonable non-exaggerated range
ax2.grid(True, linestyle="--")

# Custom Legend
handles, labels = ax1.get_legend_handles_labels()
# Clean up duplicate labels and organize them
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2)

plt.tight_layout()
out_path_png = Path("/Users/michael/.gemini/antigravity-ide/brain/716c4b63-13d6-488c-ab32-0e2963e773c8/tradeoff_plot.png")
out_path_pdf = Path("/Users/michael/.gemini/antigravity-ide/brain/716c4b63-13d6-488c-ab32-0e2963e773c8/tradeoff_plot.pdf")
out_path_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
fig.savefig(out_path_pdf, format="pdf", bbox_inches="tight")
print(f"Plot saved to {out_path_png} and {out_path_pdf}")
