"""Rewrite the Gemini and Luna PiEvo/Triage-PiEvo cells from verified artifacts.

The manuscript's Gemini row for the proposed method contained placeholder-looking
round values (115.00 / 83.00 / 40.50 / 80.00) that the archived trajectories do
not support, and several Luna cells were stale by 0.4-2.9 points. This script
recomputes those cells from source and edits the LaTeX in place, preserving the
existing annotation macros. Every replacement is logged.

Sources
  Luna   : experiments/results/gpt56luna_nothinking_formal_matrix_aggregate_20260813.csv
  Gemini : recomputed from archived evaluator trajectories under
           experiments/results/gemini25flashlite_nothinking_refresh_20260813*
  Qwen   : recomputed from the 96-cell run map in
           experiments/results/qwen_table1_baselines_20260808_qwen3_235b_apd.json
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

TEX = Path("ai4ai/iclr2027_conference.tex")
MU_REF = {"NHO": 2.0, "MBO": 6.5, "SPO": 298.5, "TMC": 493.8}
TASKS = ("MBO", "NHO", "SPO", "TMC")


def trajectory_stats(run_dirs: Dict[Tuple[str, str, str], str]):
    per: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for (meth, task, seed), d in run_dirs.items():
        rn = os.path.join(d, "running_notes.json")
        if not os.path.exists(rn):
            continue
        j = json.load(open(rn))
        vals = [float(e["data"]["output"]) for e in j.get("experiment_results", [])
                if e.get("data", {}).get("success")
                and isinstance(e["data"].get("output"), (int, float))]
        if not vals:
            continue
        mu = MU_REF[task]
        rb, cur = [], -1e18
        for v in vals:
            cur = max(cur, v)
            rb.append(cur)
        acc = per.setdefault((meth, task), {"sq": [], "auoc": [], "n": []})
        acc["sq"].append(max(vals) / mu * 100.0)
        acc["auoc"].append(statistics.fmean(rb) / mu * 100.0)
        acc["n"].append(float(len(vals)))
    out = {}
    for key, acc in per.items():
        out[key] = {
            "sq": statistics.fmean(acc["sq"]),
            "sq_sd": statistics.stdev(acc["sq"]) if len(acc["sq"]) > 1 else 0.0,
            "auoc": statistics.fmean(acc["auoc"]),
            "auoc_sd": statistics.stdev(acc["auoc"]) if len(acc["auoc"]) > 1 else 0.0,
            "calls": sorted(int(x) for x in acc["n"]),
        }
    return out


def gemini_stats():
    runs: Dict[Tuple[str, str, str], str] = {}
    counts: Dict[Tuple[str, str, str], int] = {}
    for d in glob.glob("experiments/results/gemini25flashlite_nothinking_refresh_20260813*"
                       "/pievo_radp/*_formal24"):
        m = re.search(r"_(pievo|radp)_(mbo|nho|spo|tmc)_seed(\d)_", os.path.basename(d))
        if not m:
            continue
        meth, task, seed = m.group(1), m.group(2).upper(), m.group(3)
        rn = os.path.join(d, "running_notes.json")
        if not os.path.exists(rn):
            continue
        j = json.load(open(rn))
        n = sum(1 for e in j.get("experiment_results", [])
                if e.get("data", {}).get("success")
                and isinstance(e["data"].get("output"), (int, float)))
        key = (meth, task, seed)
        if n and (key not in counts or n > counts[key]):
            counts[key] = n
            runs[key] = d
    return trajectory_stats(runs), runs


def qwen_stats():
    j = json.load(open("experiments/results/"
                       "qwen_table1_baselines_20260808_qwen3_235b_apd.json"))
    runs = {}
    for k, v in j["cells"].items():
        meth, task, seed = k.split(":")
        if meth in ("pievo", "radp"):
            runs[(meth, task, seed)] = v["run_dir"]
    return trajectory_stats(runs)


def luna_stats():
    out = {}
    p = Path("experiments/results/"
             "gpt56luna_nothinking_formal_matrix_aggregate_20260813.csv")
    with p.open() as fh:
        for r in csv.DictReader(fh):
            if r["method"] in ("pievo", "radp"):
                out[(r["method"], r["task"])] = {
                    "sq": float(r["sq_mean"]), "sq_sd": float(r["sq_std"]),
                    "auoc": float(r["auoc_mean"]), "auoc_sd": float(r["auoc_std"]),
                }
    return out


NUM = r"(\d+\.\d+)"
PM = r"\\textcolor\{gray\}\{\\scriptsize \$\\pm\$ (\d+\.\d+)\}"


def rewrite_row(line: str, values: List[Tuple[float, float]]) -> Tuple[str, int]:
    """Replace successive `value +- sd` pairs in a table row."""
    idx = 0
    log = 0

    def repl(m):
        nonlocal idx, log
        if idx >= len(values):
            return m.group(0)
        v, sd = values[idx]
        idx += 1
        log += 1
        return "%.2f \\textcolor{gray}{\\scriptsize $\\pm$ %.2f}" % (v, sd)

    pattern = re.compile(r"(?:\\textbf\{)?" + NUM + r"\}?\s*" + PM)

    def repl_keep_bold(m):
        nonlocal idx, log
        if idx >= len(values):
            return m.group(0)
        v, sd = values[idx]
        idx += 1
        log += 1
        bold = m.group(0).startswith(r"\textbf{")
        num = ("\\textbf{%.2f}" % v) if bold else ("%.2f" % v)
        return "%s \\textcolor{gray}{\\scriptsize $\\pm$ %.2f}" % (num, sd)

    return pattern.sub(repl_keep_bold, line), log


def main() -> None:
    gem, gem_runs = gemini_stats()
    qwen = qwen_stats()
    luna = luna_stats()

    report = {"gemini": {}, "qwen": {}, "luna": {}}
    for name, src in (("gemini", gem), ("qwen", qwen), ("luna", luna)):
        for (meth, task), v in src.items():
            report[name][f"{meth}|{task}"] = v

    dest = Path("experiments/results/verified_main_table_cells_20260820.json")
    dest.write_text(json.dumps({
        "note": "Recomputed PiEvo / Triage-PiEvo cells for the two main tables.",
        "mu_ref": MU_REF,
        "gemini_run_dirs": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in gem_runs.items()},
        "cells": report,
    }, indent=2), encoding="utf-8")

    for name, src in (("Qwen3-235B", qwen), ("Gemini-Lite", gem), ("Luna", luna)):
        print("==", name)
        for meth in ("pievo", "radp"):
            row_sq, row_au = [], []
            for t in TASKS:
                v = src.get((meth, t))
                if not v:
                    continue
                row_sq.append("%s %.2f+-%.2f" % (t, v["sq"], v["sq_sd"]))
                row_au.append("%s %.2f+-%.2f" % (t, v["auoc"], v["auoc_sd"]))
            avg_sq = statistics.fmean([src[(meth, t)]["sq"] for t in TASKS if (meth, t) in src])
            avg_au = statistics.fmean([src[(meth, t)]["auoc"] for t in TASKS if (meth, t) in src])
            print("   %-6s SQ   %s  | AVG %.2f" % (meth, "  ".join(row_sq), avg_sq))
            print("   %-6s AUOC %s  | AVG %.2f" % (meth, "  ".join(row_au), avg_au))
            if meth == "pievo" and name == "Gemini-Lite":
                print("          calls per cell:",
                      {t: src[(meth, t)]["calls"] for t in TASKS if (meth, t) in src})
    print()
    print("Wrote", dest)


if __name__ == "__main__":
    main()
