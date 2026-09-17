#!/usr/bin/env bash
set -euo pipefail

# Explicit model-track wrapper.  It keeps the released PiEvo/RADP runner and
# evaluator accounting, while making the Luna model and medium reasoning
# setting impossible to omit accidentally.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIEVO_AGENT_MODEL="${PIEVO_AGENT_MODEL:-gpt-5.6-luna}"
export PIEVO_MODEL_TRACK="gpt56luna"
export PIEVO_REASONING_EFFORT="medium"
export PIEVO_OUTCOME_BLIND_RECOVERY="1"
export PIEVO_RECOVERY_CONTRACT_ID="outcome_blind_structural_recovery_v2"
export PIEVO_PROTOCOL_PATH="${PIEVO_PROTOCOL_PATH:-$ROOT/experiments/configs/gpt56luna_medium_pievo_radp_protocol_v1.json}"
export PIEVO_DECODE_PROTOCOL="${PIEVO_DECODE_PROTOCOL:-$ROOT/experiments/configs/gpt56luna_medium_pievo_radp_protocol_v1.json}"
if [[ "${PIEVO_RUN_STAGE:-formal}" == "development" ]]; then
  export PIEVO_RUN_TRACK="${PIEVO_RUN_TRACK:-gpt56luna_reconstruction}"
else
  export PIEVO_RUN_TRACK="${PIEVO_RUN_TRACK:-gpt56luna}"
fi
exec "$ROOT/experiments/run_matched_qwen235b_task.sh" "$@"
