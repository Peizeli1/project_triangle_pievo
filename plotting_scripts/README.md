# Triage-PiEvo — Figure Generation Scripts

Each script regenerates exactly one figure that appears in the paper
(`main` branch). File names match the figure name used in the paper's
LaTeX source (`figures/<name>.pdf`).

| Script | Produces | Paper location |
|---|---|---|
| `fig_afe_manifold.py` | `fig_afe_manifold.pdf` | Figure 2 — AFE portfolio (schematic) |
| `fig_budget_pareto.py` | `fig_budget_pareto.pdf` | Appendix Figure — performance progression & cost-effectiveness Pareto frontier |
| `fig_gate_sensitivity.py` | `fig_gate_sensitivity.pdf` | Appendix Figure — gate sensitivity analysis (needs `data/gate_sensitivity_20260820/gate_sensitivity.json`, included) |
| `fig_runtime.py` | `fig_runtime.pdf` | Appendix Figure — execution runtime & discovery throughput across baselines |
| `fig_tradeoff_plugin.py` | `fig_tradeoff_plugin.pdf` | Figure 5 — cost-efficiency trade-offs across autonomous research harnesses |

## Usage

Each script runs standalone and writes its output PDF (and PNG where
applicable) into the current working directory:

```bash
cd plotting_scripts
python3 fig_afe_manifold.py
python3 fig_budget_pareto.py
python3 fig_gate_sensitivity.py
python3 fig_runtime.py
python3 fig_tradeoff_plugin.py
```

Dependencies: `matplotlib`, `numpy`.

## Note

Two figures that appear in the paper (`fig_method_overview.pdf` and
`triage-overview.pdf`) are hand-drawn schematics with no generating
script and are not included here.
