#!/usr/bin/env python3
"""Publication-ready Case Studies figure for ICLR 2027.

Grounding in paper:
  - Case 1: Molecular Bio-activity Optimization (MBO) under evaluator anomaly / corruption (Section 4.3 & Table 4).
            Candidate M_14 receives anomalous output y_14 = 8.40 vs expected 6.56 (surprisal 0.97 > tau_anom = 0.85).
            PiEvo accepts false principle ("prioritize dense sp3 polycycles"), derailing search (100% corrupt admission).
            Triage-PiEvo isolates into Provisional, verifies disagreement, Quarantines & Replaces (0% corrupt admission, 100% replaced).
  - Case 2: Reproducible Local Artifact vs. Generalizing Mechanism & Recursive Backtracking (Section 4.4 & Section 4.5).
            Same-source repetition reproduces artifact (fooled naive verifier, 1.0 false expansion).
            Triage requires k=2 independent held-out records; catches non-generalizing artifact.
            Upon mechanism reversal, executes recursive retraction (100% retraction accuracy vs 0% for baseline).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_card(ax, xy, width, height, title, lines, title_bg="#ECEFF1", border_color="#B0BEC5",
              title_color="#263238", text_color="#37474F", title_fontsize=6.6, text_fontsize=5.4,
              lw=0.8, box_bg="#FFFFFF"):
    """Draw a clean card box with a distinct header band and well-padded text."""
    x, y = xy
    # Base container
    rect = patches.FancyBboxPatch((x, y), width, height,
                                  boxstyle="round,pad=0.005,rounding_size=0.02",
                                  facecolor=box_bg, edgecolor=border_color, linewidth=lw, zorder=2)
    ax.add_patch(rect)
    # Header band
    header_h = height * 0.22
    header = patches.FancyBboxPatch((x, y + height - header_h), width, header_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.02",
                                    facecolor=title_bg, edgecolor=border_color, linewidth=lw, zorder=3)
    ax.add_patch(header)
    # Title text
    ax.text(x + width / 2.0, y + height - header_h / 2.0, title,
            ha="center", va="center", fontsize=title_fontsize, fontweight="bold",
            color=title_color, zorder=4)
    # Body lines
    n = len(lines)
    available_h = height - header_h - 0.02
    dy = available_h / max(1, n)
    for i, line in enumerate(lines):
        line_y = y + height - header_h - 0.012 - (i + 0.5) * dy
        ax.text(x + 0.012, line_y, line,
                ha="left", va="center", fontsize=text_fontsize, color=text_color, zorder=4)


def draw_pill(ax, xy, width, height, text, bg_color="#E0E0E0", text_color="#212121",
              fontsize=6.2, border_color="#BDBDBD", bold=True):
    """Draw a compact status pill."""
    x, y = xy
    pill = patches.FancyBboxPatch((x, y), width, height,
                                  boxstyle="round,pad=0.004,rounding_size=0.02",
                                  facecolor=bg_color, edgecolor=border_color, linewidth=0.7, zorder=4)
    ax.add_patch(pill)
    ax.text(x + width / 2.0, y + height / 2.0, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=text_color, zorder=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("ai4ai/figures"))
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.1, 5.6), gridspec_kw={"height_ratios": [1.0, 1.0]})

    # =========================================================================
    # Case Study 1: Molecular Anomaly Quarantining (MBO Chemical Task)
    # =========================================================================
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # Outer container
    frame1 = patches.FancyBboxPatch((0.005, 0.01), 0.99, 0.98,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    facecolor="#F8FAFC", edgecolor="#CFD8DC", linewidth=0.8, zorder=1)
    ax1.add_patch(frame1)

    ax1.text(0.02, 0.93, "(a) Case Study 1: Molecular Bio-activity (MBO) — Evaluator Anomaly & False Principle Revision",
             fontsize=8.2, fontweight="bold", color="#1A237E")

    # Column 1: Candidate Observation Card (x: 0.02 -> 0.25)
    draw_card(ax1, (0.02, 0.10), 0.23, 0.78,
              "Evaluation at Step 14",
              ["• Task: Bio-activity (MBO)",
               "• Molecule: Dense polycycle M_14",
               "• Observed y_14: 8.40 (Spike)",
               "• Model prediction μ̂: 6.56",
               "• Surprisal s_t: 0.97 > 0.85 (Anom!)",
               "• Latent cause: Transient noise /",
               "  surrogate evaluation mismatch",
               "• Dilemma: Mechanism vs Error?"],
              title_bg="#E3F2FD", border_color="#90CAF9", title_color="#0D47A1", text_fontsize=5.4)

    # Connecting arrows between Col 1 and Col 2 (gap 0.25 to 0.355)
    ax1.annotate("", xy=(0.355, 0.68), xytext=(0.252, 0.68),
                 arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2, shrinkA=2, shrinkB=2), zorder=3)
    ax1.text(0.303, 0.71, "Direct Revise", ha="center", va="bottom", fontsize=5.8,
             fontweight="bold", color="#C62828", zorder=5)

    ax1.annotate("", xy=(0.355, 0.30), xytext=(0.252, 0.30),
                 arrowprops=dict(arrowstyle="->", color="#00897B", lw=1.2, shrinkA=2, shrinkB=2), zorder=3)
    ax1.text(0.303, 0.33, "Source Triage", ha="center", va="bottom", fontsize=5.8,
             fontweight="bold", color="#00695C", zorder=5)

    # Column 2 (Upper): Baseline PiEvo Action (x: 0.355 -> 0.655)
    draw_card(ax1, (0.355, 0.50), 0.30, 0.38,
              "Baseline (PiEvo): Direct Revision",
              ["• Treats raw residual as scientific novelty",
               "• Updates principle space: P_t → P_t+1",
               "• Rule: 'Heavily favor rigid polycycles'",
               "• Unconditionally admitted to trusted state"],
              title_bg="#FFCDD2", border_color="#E57373", title_color="#B71C1C", text_fontsize=5.4)

    # Connecting arrow between Col 2 and Col 3
    ax1.annotate("", xy=(0.69, 0.68), xytext=(0.657, 0.68),
                 arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2, shrinkA=2, shrinkB=2), zorder=3)

    # Column 3 (Upper): Baseline Failure Outcome (x: 0.69 -> 0.98)
    draw_card(ax1, (0.69, 0.50), 0.29, 0.38,
              "Downstream Failure Outcome",
              ["• Search misled into sub-optimal dead end",
               "• Corrupt admission rate: 100% (Table 4)",
               "• Proxy-best contamination: 100%",
               "• Budget wasted; SQ drops to 70.35%"],
              title_bg="#FFEBEE", border_color="#EF9A9A", title_color="#C62828", text_fontsize=5.4)

    # Column 2 (Lower): Triage-PiEvo Action
    draw_card(ax1, (0.355, 0.10), 0.30, 0.38,
              "Triage-PiEvo: Gated Admission",
              ["• Opens diagnostic state; Z_t treated latent",
               "• Principle quarantined as PROVISIONAL",
               "• Verification query: v_t^rep = 5.90 ≠ 8.40",
               "• Disagreement → Quarantined & Replaced"],
              title_bg="#B2DFDB", border_color="#4DB6AC", title_color="#004D40", text_fontsize=5.4)

    # Connecting arrow between Col 2 and Col 3
    ax1.annotate("", xy=(0.69, 0.30), xytext=(0.657, 0.30),
                 arrowprops=dict(arrowstyle="->", color="#00897B", lw=1.2, shrinkA=2, shrinkB=2), zorder=3)

    # Column 3 (Lower): Triage-PiEvo Protected Outcome
    draw_card(ax1, (0.69, 0.10), 0.29, 0.38,
              "Protected Discovery Outcome",
              ["• Corrupt admission rate: 0.0% (Table 4)",
               "• Corrupted evidence replaced: 100%",
               "• Nominal trajectory safely preserved",
               "• Discovers true optimal lead (SQ: 132.0%)"],
              title_bg="#C8E6C9", border_color="#81C784", title_color="#1B5E20", text_fontsize=5.4)

    # =========================================================================
    # Case Study 2: Local Artifact vs Generalizing Mechanism & Backtracking
    # =========================================================================
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    frame2 = patches.FancyBboxPatch((0.005, 0.01), 0.99, 0.98,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    facecolor="#FCFDFD", edgecolor="#CFD8DC", linewidth=0.8, zorder=1)
    ax2.add_patch(frame2)

    ax2.text(0.02, 0.93, "(b) Case Study 2: Reproducible Local Artifact vs. Generalizing Mechanism & Recursive Backtracking",
             fontsize=8.2, fontweight="bold", color="#E65100")

    # Column 1: Local Artifact Dilemma Card (x: 0.02 -> 0.26)
    draw_card(ax2, (0.02, 0.10), 0.24, 0.78,
              "1. Multi-Channel Evidence",
              ["• Primary Channel: y_t = 7.85",
               "• Same-Channel Repeat: 7.82 (Agrees!)",
               "    → Artifact reproduces locally",
               "• Held-out Probe: 5.20 (Contradicts)",
               "    → Held-out gain < τ_gen (0.16)",
               "• Dilemma: Reproducible locally,",
               "    fails independent generalization",
               "• Mechanism reverses at step 36"],
              title_bg="#FFF8E1", border_color="#FFE082", title_color="#F57F17", text_fontsize=5.4)

    # Column 2: Principle Lifecycle State Machine (x: 0.285 -> 0.525)
    rect_state = patches.FancyBboxPatch((0.285, 0.08), 0.24, 0.80,
                                        boxstyle="round,pad=0.008,rounding_size=0.02",
                                        facecolor="#FFFFFF", edgecolor="#B0BEC5", linewidth=0.7, zorder=2)
    ax2.add_patch(rect_state)
    ax2.text(0.405, 0.84, "Principle Lifecycle", ha="center", fontsize=7.0,
             fontweight="bold", color="#37474F", zorder=3)

    draw_pill(ax2, (0.315, 0.69), 0.18, 0.085, "PROPOSED", bg_color="#ECEFF1", text_color="#455A64", border_color="#CFD8DC", fontsize=6.2)
    draw_pill(ax2, (0.315, 0.49), 0.18, 0.085, "PROVISIONAL", bg_color="#FFF9C4", text_color="#F57F17", border_color="#FFF176", fontsize=6.2)
    draw_pill(ax2, (0.315, 0.29), 0.18, 0.085, "ACCEPTED", bg_color="#C8E6C9", text_color="#2E7D32", border_color="#A5D6A7", fontsize=6.2)
    draw_pill(ax2, (0.315, 0.11), 0.18, 0.085, "RETRACTED", bg_color="#FFCDD2", text_color="#C62828", border_color="#EF9A9A", fontsize=6.2)

    # Vertical arrows with text beside arrow (not overlapping)
    ax2.annotate("", xy=(0.405, 0.58), xytext=(0.405, 0.685),
                 arrowprops=dict(arrowstyle="->", color="#78909C", lw=1.0, shrinkA=2, shrinkB=2), zorder=3)
    ax2.text(0.418, 0.635, "1st Support", ha="left", va="center", fontsize=5.5, color="#546E7A", fontweight="bold")

    ax2.annotate("", xy=(0.405, 0.38), xytext=(0.405, 0.485),
                 arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.0, shrinkA=2, shrinkB=2), zorder=3)
    ax2.text(0.418, 0.435, "k ≥ 2 Held-out", ha="left", va="center", fontsize=5.5, color="#2E7D32", fontweight="bold")

    ax2.annotate("", xy=(0.405, 0.20), xytext=(0.405, 0.285),
                 arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.0, shrinkA=2, shrinkB=2), zorder=3)
    ax2.text(0.418, 0.245, "Contradiction", ha="left", va="center", fontsize=5.5, color="#C62828", fontweight="bold")

    # Column 3 (Upper): Naive / Repeat-Only Verifier (x: 0.55 -> 0.98)
    draw_card(ax2, (0.55, 0.50), 0.43, 0.38,
              "Naive / Repeat-Only Verifier",
              ["• Conflates repeatability with true generalizability",
               "• Promotes non-generalizing artifact to Accepted state",
               "• False expansion rate on local artifacts: 1.0 (Section 4.4)",
               "• Mechanism Reversal: Lacks recursive retraction mechanism",
               "• Retraction accuracy: 0.0% → Knowledge permanently corrupted"],
              title_bg="#FFEBEE", border_color="#EF9A9A", title_color="#C62828", text_fontsize=5.4)

    # Column 3 (Lower): Triage-PiEvo with Backtracking
    draw_card(ax2, (0.55, 0.10), 0.43, 0.38,
              "Triage-PiEvo: Held-Out Gating & Backtracking",
              ["• Held-out probe fails gain threshold (gain < 0.16) → Keeps Provisional",
               "• False expansion rate on local artifacts: 0.0% (Section 4.4)",
               "• Zero false mechanism revisions admitted to persistent state",
               "• Mechanism Reversal: Executes recursive backtracking of tree",
               "• Retraction accuracy: 1.000 (100% Correct Final State Recovery)"],
              title_bg="#E8F5E9", border_color="#A5D6A7", title_color="#1B5E20", text_fontsize=5.4)

    fig.tight_layout(pad=0.3)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.outdir / "fig_case_study.pdf"
    png_path = args.outdir / "fig_case_study.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
