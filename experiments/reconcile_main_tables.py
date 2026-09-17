"""Reconcile every numeric cell of the two main tables against source artifacts.

Purpose: the manuscript tables must be regenerable from archived results. This
script recomputes each (backbone, method, task) SQ/AUOC cell from the frozen
aggregate artifacts and reports any cell that the artifacts cannot support.

It deliberately does not "fix" the manuscript. It classifies each cell as:
  MATCH     value agrees with an artifact within tolerance
  MISMATCH  an artifact exists but disagrees
  NO-SOURCE no artifact backs the value
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
from pathlib import Path
from typing import Dict, Optional, Tuple

MU_REF = {"NHO": 2.0, "MBO": 6.5, "SPO": 298.5, "TMC": 493.8}
TASKS = ("MBO", "NHO", "SPO", "TMC")
TOL = 0.05

LUNA_CSV = Path("experiments/results/gpt56luna_nothinking_formal_matrix_aggregate_20260813.csv")
GEMINI_GLOB = ("experiments/results/gemini25flashlite_nothinking_refresh_20260813*"
               "/pievo_radp/*_formal24")


def luna_aggregate() -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not LUNA_CSV.exists():
        return out
    with LUNA_CSV.open() as fh:
        for r in csv.DictReader(fh):
            out[(r["method"], r["task"])] = {
                "sq": float(r["sq_mean"]), "sq_sd": float(r["sq_std"]),
                "auoc": float(r["auoc_mean"]), "auoc_sd": float(r["auoc_std"]),
            }
    return out


def gemini_recomputed() -> Dict[Tuple[str, str], Dict[str, object]]:
    """Recompute Gemini cells directly from archived evaluator trajectories."""
    best: Dict[Tuple[str, str, str], list] = {}
    for d in glob.glob(GEMINI_GLOB):
        m = re.search(r"_(pievo|radp)_(mbo|nho|spo|tmc)_seed(\d)_", os.path.basename(d))
        if not m:
            continue
        meth, task, seed = m.groups()
        rn = os.path.join(d, "running_notes.json")
        if not os.path.exists(rn):
            continue
        j = json.load(open(rn))
        vals = [float(e["data"]["output"]) for e in j.get("experiment_results", [])
                if e.get("data", {}).get("success")
                and isinstance(e["data"].get("output"), (int, float))]
        if not vals:
            continue
        key = (meth, task, seed)
        if key not in best or len(vals) > len(best[key]):
            best[key] = vals

    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    per: Dict[Tuple[str, str], Dict[str, list]] = {}
    for (meth, task, seed), vals in best.items():
        mu = MU_REF[task.upper()]
        rb, cur = [], -1e18
        for v in vals:
            cur = max(cur, v)
            rb.append(cur)
        acc = per.setdefault((meth, task.upper()), {"sq": [], "auoc": [], "n": []})
        acc["sq"].append(max(vals) / mu * 100.0)
        acc["auoc"].append(statistics.fmean(rb) / mu * 100.0)
        acc["n"].append(len(vals))
    for key, acc in per.items():
        out[key] = {
            "sq": statistics.fmean(acc["sq"]),
            "sq_sd": statistics.stdev(acc["sq"]) if len(acc["sq"]) > 1 else 0.0,
            "auoc": statistics.fmean(acc["auoc"]),
            "auoc_sd": statistics.stdev(acc["auoc"]) if len(acc["auoc"]) > 1 else 0.0,
            "primary_calls": sorted(acc["n"]),
            "seeds": len(acc["sq"]),
        }
    return out


def parse_manuscript(tex: str, label: str, ncols: int):
    body = tex.split("label{%s}" % label)[1]
    body = body.split(r"\end{table*}")[0].split(r"\end{table}")[0]
    blocks, cur = {}, None
    for line in body.split(r"\\"):
        if "multicolumn" in line and any(t in line for t in ("Qwen", "Gemini", "GPT")):
            m = re.search(r"textbf\{([^}]*)\}", line)
            cur = m.group(1) if m else "?"
            blocks[cur] = {}
            continue
        cells = line.split("&")
        if len(cells) != ncols + 1 or cur is None:
            continue
        raw = cells[0]
        if r"\radp" in raw:
            name = "radp"
        elif r"\pievo" in raw:
            name = "pievo"
        else:
            name = re.sub(r"\\(textbf|textsc)|[{}\\]|midrule", "", raw).strip()
        if not name or name.lower().startswith("model"):
            continue
        vals = []
        for c in cells[1:]:
            c2 = re.sub(r"\\textcolor\{gray\}.*?\}\}", "", c)
            mm = re.search(r"(\d+\.\d+)", c2)
            vals.append(float(mm.group(1)) if mm else None)
        blocks[cur][name] = vals
    return blocks


def classify(paper: Optional[float], source: Optional[float]) -> str:
    if paper is None:
        return "n/a"
    if source is None:
        return "NO-SOURCE"
    return "MATCH" if abs(paper - source) <= TOL else "MISMATCH"


def main() -> None:
    tex = Path("ai4ai/iclr2027_conference.tex").read_text()
    sq_tbl = parse_manuscript(tex, "tab:main-sq", 5)
    ap_tbl = parse_manuscript(tex, "tab:main-apd-auoc", 10)

    luna = luna_aggregate()
    gem = gemini_recomputed()

    findings = []
    print("=" * 78)
    print("GPT-5.6-Luna  (source: %s)" % LUNA_CSV.name)
    print("=" * 78)
    luna_block = next((k for k in sq_tbl if "GPT" in k), None)
    for meth in ("pievo", "radp"):
        for i, task in enumerate(TASKS):
            paper = sq_tbl[luna_block].get(meth, [None] * 5)[i]
            src = luna.get((meth, task), {}).get("sq")
            st = classify(paper, src)
            if st != "MATCH":
                findings.append(("Luna", meth, task, "SQ", paper, src, st))
            print("  %-6s %-4s SQ    paper=%8s  artifact=%8s  %s" % (
                meth, task, paper, "%.2f" % src if src else None, st))

    ap_block = next((k for k in ap_tbl if "GPT" in k), None)
    for meth in ("pievo", "radp"):
        for i, task in enumerate(TASKS):
            paper = ap_tbl[ap_block].get(meth, [None] * 10)[2 * i + 1]
            src = luna.get((meth, task), {}).get("auoc")
            st = classify(paper, src)
            if st != "MATCH":
                findings.append(("Luna", meth, task, "AUOC", paper, src, st))
            print("  %-6s %-4s AUOC  paper=%8s  artifact=%8s  %s" % (
                meth, task, paper, "%.2f" % src if src else None, st))

    print()
    print("=" * 78)
    print("Gemini-2.5-Flash-Lite  (recomputed from archived trajectories)")
    print("=" * 78)
    gb = next((k for k in sq_tbl if "Gemini" in k), None)
    gab = next((k for k in ap_tbl if "Gemini" in k), None)
    for meth in ("pievo", "radp"):
        for i, task in enumerate(TASKS):
            rec = gem.get((meth, task), {})
            paper = sq_tbl[gb].get(meth, [None] * 5)[i]
            src = rec.get("sq")
            st = classify(paper, src)
            if st != "MATCH":
                findings.append(("Gemini", meth, task, "SQ", paper, src, st))
            print("  %-6s %-4s SQ    paper=%8s  recomputed=%8s  %-9s calls=%s" % (
                meth, task, paper, "%.2f" % src if src else None, st,
                rec.get("primary_calls")))
    for meth in ("pievo", "radp"):
        for i, task in enumerate(TASKS):
            rec = gem.get((meth, task), {})
            paper = ap_tbl[gab].get(meth, [None] * 10)[2 * i + 1]
            src = rec.get("auoc")
            st = classify(paper, src)
            if st != "MATCH":
                findings.append(("Gemini", meth, task, "AUOC", paper, src, st))
            print("  %-6s %-4s AUOC  paper=%8s  recomputed=%8s  %s" % (
                meth, task, paper, "%.2f" % src if src else None, st))

    print()
    print("=" * 78)
    print("SUMMARY: %d cells需要处理" % len(findings))
    print("=" * 78)
    for f in findings:
        print("  %-7s %-6s %-4s %-5s paper=%-8s source=%-8s %s" % (
            f[0], f[1], f[2], f[3], f[4],
            ("%.2f" % f[5]) if f[5] is not None else "None", f[6]))

    Path("experiments/results/table_reconciliation_20260820.json").write_text(
        json.dumps({
            "tolerance": TOL,
            "gemini_recomputed": {f"{k[0]}|{k[1]}": v for k, v in gem.items()},
            "luna_artifact": {f"{k[0]}|{k[1]}": v for k, v in luna.items()},
            "findings": [
                {"backbone": f[0], "method": f[1], "task": f[2], "metric": f[3],
                 "paper": f[4], "source": f[5], "status": f[6]} for f in findings
            ],
        }, indent=2), encoding="utf-8")
    print()
    print("Wrote experiments/results/table_reconciliation_20260820.json")


if __name__ == "__main__":
    main()
