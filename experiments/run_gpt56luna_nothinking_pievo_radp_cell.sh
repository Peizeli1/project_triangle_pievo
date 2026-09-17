#!/usr/bin/env bash
set -euo pipefail

# Dedicated no-thinking launcher.  It intentionally overrides any inherited
# reasoning setting so a no-thinking cell cannot be mislabeled as medium.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIEVO_AGENT_MODEL="gpt-5.6-luna"
export PIEVO_MODEL_TRACK="gpt56luna_nothinking"
export PIEVO_REASONING_EFFORT="none"
export PIEVO_OUTCOME_BLIND_RECOVERY="1"
export PIEVO_RECOVERY_CONTRACT_ID="outcome_blind_structural_recovery_v2"
export PIEVO_PROTOCOL_PATH="${PIEVO_PROTOCOL_PATH:-$ROOT/experiments/configs/gpt56luna_nothinking_pievo_radp_protocol_v1.json}"
export PIEVO_DECODE_PROTOCOL="${PIEVO_DECODE_PROTOCOL:-$ROOT/experiments/configs/gpt56luna_nothinking_pievo_radp_protocol_v1.json}"
if [[ "${PIEVO_RUN_STAGE:-formal}" == "development" ]]; then
  export PIEVO_RUN_TRACK="gpt56luna_nothinking_reconstruction"
else
  export PIEVO_RUN_TRACK="gpt56luna_nothinking"
  export PIEVO_FORMAL_APPROVAL="${PIEVO_FORMAL_APPROVAL:-$ROOT/experiments/results/formal_approvals/${PIEVO_MODEL_TRACK}_${PIEVO_METHOD}_${PIEVO_TASK,,}_v2.json}"
  export PIEVO_MODEL_TRACK_FREEZE="${PIEVO_MODEL_TRACK_FREEZE:-$ROOT/experiments/results/${PIEVO_MODEL_TRACK}_${PIEVO_METHOD}_${PIEVO_TASK,,}_protocol_freeze_v2.json}"
fi
exec "$ROOT/experiments/run_matched_qwen235b_task.sh" "$@"
