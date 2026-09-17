# Triage-PiEvo — Code

This branch contains the experiment code for Triage-PiEvo. The paper
source and compiled PDF live on `main`.

## Layout

- `experiments/` — top-level runners, configs, and analysis scripts
  (`.py`/`.sh`/`.yaml`/`.json`) used to produce the results reported
  in the paper.
- `experiments/configs/`, `experiments/qwen_required_suite/`,
  `experiments/sciagentarena_formal/` — task configs and per-suite
  runners.
- `experiments/external_harness_pilot/` — self-authored cross-harness
  pilot code (`stateful_triage/`, `AARRI-pilot/`, three-arm runners,
  smoke tests). Third-party benchmark clones (AARRI-bench,
  ResearchClawBench) used by this pilot are **not** included here;
  clone them separately from their own repos if reproducing the
  cross-harness experiments.

## Not included

- Third-party benchmark/baseline repos (PiEvo, PiFlow, AI-Researcher,
  SciAgentArena, AARRI-bench, ResearchClawBench, etc.) — clone from
  their original sources.
- Python virtual environments.
- Raw experiment logs/databases/traces from `experiments/results/`
  (large, run-specific artifacts; kept locally / in separate storage).

## Credentials

API keys are never hardcoded. Scripts read from environment variables
or macOS Keychain (see `experiments/runtime_credentials.py` and
`experiments/external_harness_pilot/setup_researchclaw_credentials.command`).
