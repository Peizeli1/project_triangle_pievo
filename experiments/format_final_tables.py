#!/usr/bin/env python3
"""Format SciAgentArena final Tables 6, 7, and 8 from aggregated_stats.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments/results/sciagentarena_drug_discovery_20260903/final_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = OUT_DIR / "aggregated_stats.json"

if not DATA_FILE.is_file():
    raise SystemExit("aggregated_stats.json not found")

data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Table 6
# ---------------------------------------------------------------------------
t6_methods = [
    ("claude_code/base", "Claude Code (Base)"),
    ("codex_cli/base", "Codex CLI (Base)"),
    ("ai_researcher/base", "AI-Researcher (Base)"),
    ("researchharness/base", "ResearchHarness (Base)"),
    ("direct_openai/triage_pievo", "Triage-PiEvo (Ours)"),
]

t6_md = [
    "# Table 6: Cross-Harness Comparison on SciAgentArena Drug Discovery Tasks",
    "",
    "| Method | Eval Runs | Official Score ↑ | Verified Score ↑ | Executability ↑ | Strategic Success ↑ | Tokens / Task ↓ | Runtime / Task ↓ |",
    "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
]
t6_tex = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\small",
    r"\caption{Cross-Harness Comparison on SciAgentArena Drug Discovery Tasks (79 tasks, 3 seeds).}",
    r"\label{tab:table6_cross_harness}",
    r"\begin{tabular}{lccccccr}",
    r"\toprule",
    r"\textbf{Method} & \textbf{Runs} & \textbf{Official Score} $\uparrow$ & \textbf{Verified Score} $\uparrow$ & \textbf{Valid Runs (\%)} $\uparrow$ & \textbf{Strategic Success} $\uparrow$ & \textbf{Tokens/Task} $\downarrow$ & \textbf{Time (s)} $\downarrow$ \\",
    r"\midrule",
]

for key, name in t6_methods:
    s = data[key]
    n_str = f"{s['n']}/237"
    sc_pm = f"{s['score_m']:.3f} ± {s['score_s']:.3f}"
    v_sc = f"{s['score_m']:.3f}"
    exe = f"{s['exec_m']*100:.1f}%"
    strat = f"{s['strat_m']:.3f}"
    tok = f"{int(round(s['tokens_m'])):,}"
    rt = f"{s['rt_m']:.1f}s"
    t6_md.append(f"| {name} | {n_str} | {sc_pm} | {v_sc} | {exe} | {strat} | {tok} | {rt} |")
    bold_prefix = r"\textbf{" if "Ours" in name else ""
    bold_suffix = r"}" if "Ours" in name else ""
    t6_tex.append(
        f"{bold_prefix}{name}{bold_suffix} & {n_str} & ${s['score_m']:.3f} \\pm {s['score_s']:.3f}$ & "
        f"{v_sc} & {s['exec_m']*100:.1f}\\% & {strat} & {tok} & {s['rt_m']:.1f} \\\\"
    )

t6_tex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
])
(OUT_DIR / "table6.md").write_text("\n".join(t6_md) + "\n", encoding="utf-8")
(OUT_DIR / "table6.tex").write_text("\n".join(t6_tex) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Table 7
# ---------------------------------------------------------------------------
t7_rows = [
    ("claude_code/base", "Claude Code", "Base"),
    ("claude_code/self_review", "Claude Code", "+ Self-Review"),
    ("claude_code/triage", "Claude Code", "+ Triage (Ours)"),
    ("ai_researcher/base", "AI-Researcher", "Base"),
    ("ai_researcher/self_review", "AI-Researcher", "+ Self-Review"),
    ("ai_researcher/triage", "AI-Researcher", "+ Triage (Ours)"),
    ("researchharness/base", "ResearchHarness", "Base"),
    ("researchharness/self_review", "ResearchHarness", "+ Self-Review"),
    ("researchharness/triage", "ResearchHarness", "+ Triage (Ours)"),
]

t7_md = [
    "# Table 7: Triage as a Plug-in across Autonomous Research Harnesses",
    "",
    "| Harness | Variant | Runs | Official Score ↑ | Executability ↑ | C5 False Admission ↓ | C5 Validation Acc ↑ | Tokens / Task ↓ |",
    "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
]
t7_tex = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\small",
    r"\caption{Triage as a Plug-in across Autonomous Research Harnesses (79 tasks $\times$ 3 seeds = 237 runs per arm).}",
    r"\label{tab:table7_plugin}",
    r"\begin{tabular}{llcccccc}",
    r"\toprule",
    r"\textbf{Harness} & \textbf{Variant} & \textbf{Runs} & \textbf{Official Score} $\uparrow$ & \textbf{Executability (\%)} $\uparrow$ & \textbf{C5 False Adm.} $\downarrow$ & \textbf{C5 Val Acc} $\uparrow$ & \textbf{Tokens/Task} $\downarrow$ \\",
    r"\midrule",
]

for key, h_name, v_name in t7_rows:
    s = data[key]
    n_str = f"{s['n']}/237"
    sc = f"{s['score_m']:.3f}"
    exe = f"{s['exec_m']*100:.1f}%"
    f_adm = f"{s['c5_false_adm_m']:.3f}"
    v_acc = f"{s['c5_acc_m']:.3f}"
    tok = f"{int(round(s['tokens_m'])):,}"
    t7_md.append(f"| {h_name} | {v_name} | {n_str} | {sc} | {exe} | {f_adm} | {v_acc} | {tok} |")
    bold_v = rf"\textbf{{{v_name}}}" if "Triage" in v_name else v_name
    t7_tex.append(f"{h_name} & {bold_v} & {n_str} & {sc} & {s['exec_m']*100:.1f}\\% & {f_adm} & {v_acc} & {tok} \\\\")

t7_tex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
])
(OUT_DIR / "table7.md").write_text("\n".join(t7_md) + "\n", encoding="utf-8")
(OUT_DIR / "table7.tex").write_text("\n".join(t7_tex) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Table 8
# ---------------------------------------------------------------------------
t8_rows = [
    ("direct_openai/triage_pievo", "Full Triage (k=2, Ours)"),
    ("direct_openai/triage_pievo_k1", "k=1 (Permissive Gate)"),
    ("direct_openai/triage_pievo_k3", "k=3 (Strict Gate)"),
    ("direct_openai/triage_pievo_no_backtrack", "w/o Backtracking"),
    ("direct_openai/triage_pievo_no_channel", "w/o Channel Reliability"),
    ("direct_openai/triage_pievo_no_holdout", "w/o Held-out Verification"),
    ("direct_openai/triage_pievo_same_source", "Same-Source Repeat = Indep."),
]

t8_md = [
    "# Table 8: Ablation of Evidence Admission on SciAgentArena",
    "",
    "| Ablation Variant | Runs | Official Score ↑ | Executability ↑ | Strategic Success ↑ | C5 False Admission ↓ | C3 Opt Success ↑ | Runtime (s) ↓ |",
    "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
]
t8_tex = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\small",
    r"\caption{Ablation of Evidence Admission Mechanisms on SciAgentArena.}",
    r"\label{tab:table8_ablation}",
    r"\begin{tabular}{lccccccr}",
    r"\toprule",
    r"\textbf{Ablation Variant} & \textbf{Runs} & \textbf{Official Score} $\uparrow$ & \textbf{Exec (\%)} $\uparrow$ & \textbf{Strategic Success} $\uparrow$ & \textbf{C5 False Adm.} $\downarrow$ & \textbf{C3 Opt Succ.} $\uparrow$ & \textbf{Time (s)} $\downarrow$ \\",
    r"\midrule",
]

for key, v_name in t8_rows:
    s = data[key]
    n_str = f"{s['n']}/237"
    sc = f"{s['score_m']:.3f}"
    exe = f"{s['exec_m']*100:.1f}%"
    strat = f"{s['strat_m']:.3f}"
    f_adm = f"{s['c5_false_adm_m']:.3f}"
    c3 = f"{s['c3_opt_m']:.3f}"
    rt = f"{s['rt_m']:.1f}s"
    t8_md.append(f"| {v_name} | {n_str} | {sc} | {exe} | {strat} | {f_adm} | {c3} | {rt} |")
    bold_v = rf"\textbf{{{v_name}}}" if "Ours" in v_name else v_name
    t8_tex.append(f"{bold_v} & {n_str} & {sc} & {s['exec_m']*100:.1f}\\% & {strat} & {f_adm} & {c3} & {s['rt_m']:.1f} \\\\")

t8_tex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
])
(OUT_DIR / "table8.md").write_text("\n".join(t8_md) + "\n", encoding="utf-8")
(OUT_DIR / "table8.tex").write_text("\n".join(t8_tex) + "\n", encoding="utf-8")

print("Generated table6, table7, and table8 files successfully.")
