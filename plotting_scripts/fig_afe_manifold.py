#!/usr/bin/env python3
"""Generate publication-quality AFE manifold figure for ICLR 2027.

Optimized for wrapfigure layout:
- Proper aspect ratio and font scaling
- Serif font matching ICLR paper style
- Smooth Gaussian mixture contours representing the latent property landscape
- Prominent Okabe-Ito colored markers for Exploitation, Expansion, and Probe
- Annotated flow vectors demonstrating active candidate proposal dynamics
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

np.random.seed(42)
fig, ax = plt.subplots(figsize=(4.2, 3.6))

# Smooth multi-modal chemical property landscape
x = np.linspace(-3.2, 3.2, 200)
y = np.linspace(-3.0, 3.2, 200)
X, Y = np.meshgrid(x, y)

# Smooth Gaussian mixture representing objective surface + uncertainty
Z_opt = np.exp(-(X**2 + Y**2) / 1.8) * 85
Z_sub1 = np.exp(-((X - 1.6)**2 + (Y - 1.4)**2) / 1.6) * 55
Z_sub2 = np.exp(-((X + 1.6)**2 + (Y + 1.6)**2) / 1.4) * 45
Z_ridge = np.exp(-((X - 1.8)**2 + (Y + 1.5)**2) / 1.0) * 40
Z = Z_opt + Z_sub1 + Z_sub2 + Z_ridge

# Filled contours and subtle contour lines
levels = np.linspace(5, 90, 18)
cf = ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.38)
cl = ax.contour(X, Y, Z, levels=levels[::2], colors="#90CAF9", linewidths=0.5, alpha=0.5)

# 1. Role 1: Exploitation (orange circles) - tightly clustered around primary peak
expl_x = np.random.normal(0.05, 0.28, 28)
expl_y = np.random.normal(0.05, 0.28, 28)

# 2. Role 2: Expansion (blue stars) - scattered across high-uncertainty frontier
exp_angles = np.random.uniform(0, 2 * np.pi, 38)
exp_radii = np.random.uniform(1.6, 2.8, 38)
exp_x = exp_radii * np.cos(exp_angles)
exp_y = exp_radii * np.sin(exp_angles)

# 3. Role 3: Probe (green triangles) - focused along discriminating ridge/valley
probe_x = np.random.normal(1.65, 0.18, 12)
probe_y = np.random.normal(-1.45, 0.18, 12)

# Scatter points with high contrast
ax.scatter(expl_x, expl_y, c="#D55E00", s=48, marker="o", edgecolors="white",
           linewidth=0.6, label="Role 1: Exploitation (Reward Max.)", zorder=5)
ax.scatter(exp_x, exp_y, c="#0072B2", s=52, marker="*", edgecolors="white",
           linewidth=0.5, label="Role 2: Expansion (Frontier Coverage)", zorder=4)
ax.scatter(probe_x, probe_y, c="#009E73", s=46, marker="^", edgecolors="white",
           linewidth=0.6, label="Role 3: Probe (Mechanism Discr.)", zorder=5)

# Active Flow Arrows with clear labels
ax.annotate("", xy=(1.55, -1.25), xytext=(0.25, -0.15),
            arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.3, ls="--", shrinkA=3, shrinkB=3), zorder=6)
ax.text(1.05, -0.65, "Active Probe Flow", fontsize=6.2, color="#004D40",
        fontweight="bold", rotation=-38, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#80CBC4", lw=0.6, alpha=0.9))

ax.annotate("", xy=(-1.6, 1.55), xytext=(-0.25, 0.25),
            arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.3, ls="--", shrinkA=3, shrinkB=3), zorder=6)
ax.text(-1.1, 0.85, "Frontier Expansion", fontsize=6.2, color="#0D47A1",
        fontweight="bold", rotation=-44, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#90CAF9", lw=0.6, alpha=0.9))

# Titles and labels
ax.set_title("Active Flow Expansion (AFE) Latent Space", fontsize=9.0, fontweight="bold", pad=5)
ax.set_xlabel("Latent Dimension 1 (Target Similarity)", fontsize=7.8)
ax.set_ylabel("Latent Dimension 2 (Structural Novelty)", fontsize=7.8)

ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-2.8, 2.9)
ax.set_xticks([])
ax.set_yticks([])

# Clean legend
ax.legend(loc="lower left", framealpha=0.92, edgecolor="#B0BEC5", facecolor="#FAFAFA",
          handletextpad=0.4, borderpad=0.4, labelspacing=0.3)

# Aesthetics
for spine in ax.spines.values():
    spine.set_edgecolor("#78909C")
    spine.set_linewidth(0.8)

fig.tight_layout(pad=0.4)

out_dir = Path(".")
pdf_p = out_dir / "fig_afe_manifold.pdf"
png_p = out_dir / "fig_afe_manifold.png"
fig.savefig(pdf_p, bbox_inches="tight")
fig.savefig(png_p, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_p} and {png_p}")
