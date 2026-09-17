#!/usr/bin/env python3
"""Publication figure for the evidence-admission gate sensitivity analysis.

Panel (a): missed-mechanism phase map over (held-out gain threshold,
           held-out probe attenuation). Shows where the decision boundary is
           and that the constant behaves as a monotone conservatism dial.
Panel (b): verification cost vs anomaly threshold, i.e. the cost/coverage dial.

Reads only the frozen sweep artifact; no values are hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OKABE_BLUE = "#2980B9"
OKABE_VERM = "#9B59B6"
OKABE_GREEN = "#16A085"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path(__file__).parent / "data/gate_sensitivity_20260820/gate_sensitivity.json")
    ap.add_argument("--outdir", type=Path, default=Path("."))
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    pm = data["phase_map"]
    thresholds = pm["thresholds"]
    attenuations = pm["attenuations"]

    grid = np.zeros((len(thresholds), len(attenuations)))
    for i, thr in enumerate(thresholds):
        for j, a in enumerate(attenuations):
            cell = next(c for c in pm["cells"]
                        if c["heldout_gain_threshold"] == thr and c["heldout_attenuation"] == a)
            grid[i, j] = cell["missed_mechanism_rate"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.9, 2.5),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    cmap = LinearSegmentedColormap.from_list("adm", ["#F7FBFF", OKABE_BLUE, OKABE_VERM])
    im = ax0.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
    ax0.set_xticks(range(len(attenuations)))
    ax0.set_xticklabels(["%.1f" % a for a in attenuations])
    ax0.set_yticks(range(len(thresholds)))
    ax0.set_yticklabels(["%.2f" % t for t in thresholds])
    ax0.set_xlabel("Held-out probe attenuation")
    ax0.set_ylabel(r"Admission margin $\tau_{\mathrm{gen}}$")
    ax0.set_title("(a) Missed-mechanism rate", loc="left")

    # Mark the operating point used in the paper.
    default_thr = data["defaults"]["HELDOUT_GAIN_THRESHOLD"]
    if default_thr in thresholds:
        ax0.plot(0, thresholds.index(default_thr), marker="o", ms=5,
                 mfc="none", mec="black", mew=1.1, zorder=5)
        ax0.annotate("default", xy=(0, thresholds.index(default_thr)),
                     xytext=(0.55, thresholds.index(default_thr) + 0.55),
                     fontsize=6.5,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

    for i in range(len(thresholds)):
        for j in range(len(attenuations)):
            ax0.text(j, i, "%.0f" % grid[i, j], ha="center", va="center",
                     fontsize=6, color="white" if grid[i, j] > 0.5 else "#37474F")

    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03, ticks=[0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.5, width=0.5)
    cb.outline.set_linewidth(0.5)

    # Panel b: anomaly threshold vs verification cost and error rates.
    sweep = data["constant_sweeps"]["ANOMALY_THRESHOLD"]["points"]
    xs = [p["value"] for p in sweep]
    cost = [p["verification_cost"]["mean"] for p in sweep]
    miss = [p["missed_mechanism_rate"]["mean"] for p in sweep]

    ax1.plot(xs, cost, marker="o", ms=3.5, lw=1.2, color=OKABE_BLUE,
             label="Verification cost")
    ax1.set_xlabel(r"Anomaly threshold $\tau_{\mathrm{anom}}$")
    ax1.set_ylabel("Verification cost / run", color=OKABE_BLUE)
    ax1.tick_params(axis="y", labelcolor=OKABE_BLUE, width=0.5)
    ax1.tick_params(axis="x", width=0.5)
    ax1.set_ylim(-0.6, max(cost) * 1.22)

    ax2 = ax1.twinx()
    ax2.plot(xs, miss, marker="s", ms=3.5, lw=1.2, ls="--", color=OKABE_VERM,
             label="Missed mechanism")
    ax2.set_ylabel("Missed-mechanism rate", color=OKABE_VERM)
    ax2.tick_params(axis="y", labelcolor=OKABE_VERM, width=0.5)
    ax2.set_ylim(-0.06, 1.12)

    dflt = data["defaults"]["ANOMALY_THRESHOLD"]
    ax1.axvline(dflt, color="#616161", lw=0.7, ls=":", zorder=0)
    ax1.annotate("default", xy=(dflt, max(cost) * 1.05), fontsize=6.5,
                 ha="right", color="#424242", rotation=90, va="top")

    ax1.set_title("(b) Detection-coverage dial", loc="left")
    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", frameon=False,
               bbox_to_anchor=(0.02, 0.62))

    fig.tight_layout(pad=0.4)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = args.outdir / f"fig_gate_sensitivity.{ext}"
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        print("wrote", dest)


if __name__ == "__main__":
    main()
