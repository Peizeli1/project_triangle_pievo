#!/usr/bin/env python3
"""Gemini 2.5 Flash Lite convergence dataset builder, v2 — correct data."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "experiments/results/gemini_convergence_v2_20260816.json"
OUTPUT_CSV = ROOT / "experiments/results/gemini_convergence_v2_20260816.csv"
FIGURE_PDF = ROOT / "paper_iclr2027/figures/fig_gemini_convergence.pdf"
FIGURE_PNG = ROOT / "paper_iclr2027/figures/fig_gemini_convergence.png"

TASKS = ("MBO", "NHO", "SPO", "TMC")
ALL_METHODS = ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2",
               "ai_researcher", "piflow", "pievo", "radp")
LABELS = {m: m.replace("_", " ").title().replace("Ai", "AI") for m in ALL_METHODS}
LABELS["vanilla_mas"] = "Vanilla MAS"
LABELS["ai_scientist_v1"] = "AI Scientist v1"
LABELS["ai_scientist_v2"] = "AI Scientist v2"
LABELS["ai_researcher"] = "AI Researcher"
COLORS = {"vanilla_mas": "#7F7F7F", "react": "#B0B0B0", "ai_scientist_v1": "#56B4E9",
          "ai_scientist_v2": "#009E73", "ai_researcher": "#CC79A7", "piflow": "#E69F00",
          "pievo": "#0072B2", "radp": "#D55E00"}
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}
AUDIT = {"MBO": "official_chembl_tool_audit.jsonl", "NHO": "official_nho_tool_audit.jsonl",
         "SPO": "official_spo_tool_audit.jsonl", "TMC": "official_tmc_tool_audit.jsonl"}

def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))

def running_normalized(values, task):
    best = -math.inf
    result = []
    for v in values:
        best = max(best, v)
        result.append(100.0 * best / MU_REF[task])
    return result

def mstd(values):
    return {"mean": fmean(values), "std": stdev(values) if len(values) > 1 else 0.0, "n": len(values)}

def extract(path):
    rows = load_jsonl(path)
    sel = sorted(
        [r for r in rows if str(r.get("tool_name","")).startswith("evaluate_")
         and r.get("success") is True and r.get("duplicate_suppressed") is not True and finite(r.get("output"))],
        key=lambda r: int(r.get("call_index", 0)))
    indices = [int(r["call_index"]) for r in sel]
    if len(sel) != 24 or indices != list(range(1, 25)):
        raise ValueError(f"bad trajectory: {path} ({len(sel)} rows)")
    return [float(r["output"]) for r in sel]

# ---- Build run map ----
T1 = ROOT / "experiments/results/gemini25flash_nothinking_table1_20260808"
RECHECK = ROOT / "experiments/results/gemini25flash_nothinking_mbo_recheck_v6_8seed_20260809/matched"

# MBO: baselines use 3 seeds (0-2) from retry01, PiEvo/RADP use 8 seeds from recheck
MBO_BASELINES = T1 / "mbo/baselines_retry01"
MBO_BASE_OLD = T1 / "mbo/baselines"

cells = {}  # key -> (run_dir, audit_name)

def add(key, path, task):
    if path.exists():
        audit = path / AUDIT[task]
        if audit.exists():
            try:
                vals = extract(audit)
                cells[key] = (path, vals, running_normalized(vals, task))
                return True
            except ValueError:
                pass
    return False

# MBO baselines (3 seeds)
for m in ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2", "ai_researcher", "piflow"):
    for s in range(3):
        add(f"{m}:MBO:{s}", MBO_BASELINES / f"gemini25flash_nothinking_{m}_mbo_seed{s}_formal24", "MBO")

# MBO PiEvo/RADP (8 seeds from recheck)
for m in ("pievo", "radp"):
    for s in range(8):
        add(f"{m}:MBO:{s}", RECHECK / f"gemini25flash_nothinking_{m}_mbo_seed{s}_formal24", "MBO")

# MBO PiEvo/RADP fallback: check table1/mbo subdirectories
for m in ("pievo", "radp"):
    for s in range(8):
        key = f"{m}:MBO:{s}"
        if key in cells:
            continue
        for subdir in ["pievo_radp", "pievo_radp_retry05", "pievo_retry06",
                       "radp_retry06_seed0", "radp_retry06_seed12"]:
            d = T1 / f"mbo/{subdir}" / f"gemini25flash_nothinking_{m}_mbo_seed{s}_formal24"
            if add(key, d, "MBO"):
                break

# NHO/SPO/TMC: baselines
for task in ("NHO", "SPO", "TMC"):
    task_lower = task.lower()
    for base_dir_name in ["baselines_retry02", "baselines_retry01", "baselines"]:
        bdir = T1 / f"{task_lower}/{base_dir_name}"
        if bdir.exists():
            for m in ("vanilla_mas", "react", "ai_scientist_v1", "ai_scientist_v2",
                      "ai_researcher", "piflow"):
                for s in range(3):
                    add(f"{m}:{task}:{s}", bdir / f"gemini25flash_nothinking_{m}_{task_lower}_seed{s}_formal24", task)

# NHO/SPO/TMC: PiEvo + RADP
for task in ("NHO", "SPO", "TMC"):
    task_lower = task.lower()
    pr_dir = T1 / f"{task_lower}/pievo_radp"
    if pr_dir.exists():
        for m in ("pievo", "radp"):
            for s in range(3):
                add(f"{m}:{task}:{s}", pr_dir / f"gemini25flash_nothinking_{m}_{task_lower}_seed{s}_formal24", task)

# Print summary
print(f"Total cells: {len(cells)}")
by_method = {}
for key in cells:
    m, t, s = key.split(":")
    by_method.setdefault(m, []).append(f"{t}:{s}")
for m in sorted(by_method):
    entries = sorted(by_method[m])
    print(f"  {m:20s}: {len(entries)} cells")

# Check missing
missing = []
for m in ALL_METHODS:
    for t in TASKS:
        n_seeds = 8 if (t == "MBO" and m in ("pievo", "radp")) else 3
        for s in range(n_seeds):
            if f"{m}:{t}:{s}" not in cells:
                missing.append(f"{m}:{t}:{s}")
print(f"Missing: {len(missing)}")
for mk in missing[:20]:
    print(f"  {mk}")

# Aggregate
aggregates = {}
for task in TASKS:
    n_seeds = 8 if task == "MBO" else 3
    for method in ALL_METHODS:
        curves = []
        valid_seeds = []
        for s in range(8 if (task == "MBO" and method in ("pievo", "radp")) else 3):
            key = f"{method}:{task}:{s}"
            if key in cells:
                curves.append(cells[key][2])
                valid_seeds.append(s)
        if not curves:
            continue
        points = [{"budget": b + 1, **mstd([c[b] for c in curves])} for b in range(24)]
        aggregates[f"{method}:{task}"] = {"task": task, "method": method, "seeds": valid_seeds, "points": points}

# Macro-average
macro = {}
for method in ALL_METHODS:
    points = []
    for b in range(24):
        vals = []
        for task in TASKS:
            n_seeds = 8 if (task == "MBO" and method in ("pievo", "radp")) else 3
            for s in range(n_seeds):
                key = f"{method}:{task}:{s}"
                if key in cells:
                    vals.append(cells[key][2][b])
        points.append({"budget": b + 1, **mstd(vals)})
    macro[method] = points

# Print final values
print("\n=== Final values (budget 24) ===")
for task in TASKS:
    print(f"\n--- {task} ---")
    for m in ALL_METHODS:
        k = f"{m}:{task}"
        if k in aggregates:
            f = aggregates[k]["points"][-1]
            print(f"  {m:20s} = {f['mean']:7.2f} ± {f['std']:6.2f} (n={f['n']})")

# Write JSON
result = {
    "schema_version": "gemini_convergence_v2.v1",
    "model": "gemini-2.5-flash-lite", "reasoning_effort": "nothinking",
    "tasks": list(TASKS), "methods": list(ALL_METHODS),
    "metric": "normalized best-so-far SQ (%)", "normalization_reference": MU_REF,
    "aggregates": aggregates, "macro_by_method": macro,
    "outputs": {"figure_pdf": str(FIGURE_PDF), "figure_png": str(FIGURE_PNG), "csv": str(OUTPUT_CSV)},
}
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

# CSV
with OUTPUT_CSV.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scope", "task", "method", "budget", "mean_sq", "std_sq", "n"])
    for a in aggregates.values():
        for p in a["points"]:
            w.writerow(["task", a["task"], a["method"], p["budget"], f"{p['mean']:.10f}", f"{p['std']:.10f}", p["n"]])
    for m, pts in macro.items():
        for p in pts:
            w.writerow(["macro_method", "ALL", m, p["budget"], f"{p['mean']:.10f}", f"{p['std']:.10f}", p["n"]])

# Plot — 5 panels (4 tasks + macro-average), per-task y-axis
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 9,
                      "axes.labelsize": 8, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 5, figsize=(8.8, 2.95), sharey=False)
for ax, task in zip(axes, TASKS):
    for method in ALL_METHODS:
        k = f"{method}:{task}"
        if k not in aggregates:
            continue
        pts = aggregates[k]["points"]
        x = [p["budget"] for p in pts]
        mean = [float(p["mean"]) for p in pts]
        spread = [float(p["std"]) for p in pts]
        is_focus = method in ("pievo", "radp")
        is_radp = method == "radp"
        ax.plot(x, mean, color=COLORS[method],
                linewidth=2.0 if is_radp else (1.75 if is_focus else 0.95),
                linestyle="--" if is_radp else "-",
                alpha=1.0 if is_focus else 0.82,
                label=LABELS[method], zorder=5 if is_radp else (4 if is_focus else 2))
        ax.fill_between(x, [m - s for m, s in zip(mean, spread)],
                        [m + s for m, s in zip(mean, spread)],
                        color=COLORS[method], alpha=0.15 if is_radp else (0.11 if is_focus else 0.025),
                        linewidth=0, zorder=1)
    ax.set_title(task); ax.set_xlim(1, 24); ax.set_xticks((1, 6, 12, 18, 24))
    ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("Primary evaluations")
axes[0].set_ylabel("Best-so-far SQ (%)")

# Macro-average panel
ax_macro = axes[4]
for method in ALL_METHODS:
    if method not in macro:
        continue
    pts = macro[method]
    x = [p["budget"] for p in pts]
    mean = [float(p["mean"]) for p in pts]
    spread = [float(p["std"]) for p in pts]
    is_focus = method in ("pievo", "radp")
    is_radp = method == "radp"
    ax_macro.plot(x, mean, color=COLORS[method],
                  linewidth=2.0 if is_radp else (1.75 if is_focus else 0.95),
                  linestyle="--" if is_radp else "-",
                  alpha=1.0 if is_focus else 0.82,
                  label=LABELS[method], zorder=5 if is_radp else (4 if is_focus else 2))
    ax_macro.fill_between(x, [m - s for m, s in zip(mean, spread)],
                          [m + s for m, s in zip(mean, spread)],
                          color=COLORS[method], alpha=0.15 if is_radp else (0.11 if is_focus else 0.025),
                          linewidth=0, zorder=1)
ax_macro.set_title("Macro Avg")
ax_macro.set_xlim(1, 24); ax_macro.set_xticks((1, 6, 12, 18, 24))
ax_macro.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
ax_macro.spines["top"].set_visible(False); ax_macro.spines["right"].set_visible(False)
ax_macro.set_xlabel("Primary evaluations")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.025))
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.0)
FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.03)
fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print(f"\n✅ Figure saved: {FIGURE_PDF}")
print(f"   Validated cells: {len(cells)}, Missing: {len(missing)}")