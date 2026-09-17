#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL="${PIEVO_PROTOCOL_PATH:-$ROOT/experiments/configs/radp_search_afe_matched_qwen235b_protocol_v11.json}"
MODEL="${PIEVO_AGENT_MODEL:-qwen3-235b-a22b-instruct-2507}"
MODEL_TRACK="${PIEVO_MODEL_TRACK:-matched_qwen235b}"
REASONING_EFFORT="${PIEVO_REASONING_EFFORT:-}"
TASK="${PIEVO_TASK:-}"
METHOD="${PIEVO_METHOD:-pievo}"
SEED="${PIEVO_SEED:-0}"
BUDGET="${PIEVO_BUDGET:-24}"
STAGE="${PIEVO_RUN_STAGE:-formal}"
DRY_RUN="${PIEVO_DRY_RUN:-0}"

case "$TASK" in
  NHO|MBO|SPO|TMC) ;;
  *) echo "PIEVO_TASK must be one of NHO, MBO, SPO, TMC" >&2; exit 2 ;;
esac
case "$METHOD" in
  pievo|radp) ;;
  *) echo "PIEVO_METHOD must be pievo or radp" >&2; exit 2 ;;
esac
case "$STAGE" in
  development|formal) ;;
  *) echo "PIEVO_RUN_STAGE must be development or formal" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]] || [[ ! "$BUDGET" =~ ^[0-9]+$ ]]; then
  echo "PIEVO_SEED and PIEVO_BUDGET must be non-negative integers" >&2
  exit 2
fi
if [[ "${PIEVO_PREFILTER_INVALID_CANDIDATES:-0}" != "0" \
  && "${PIEVO_PREFILTER_INVALID_CANDIDATES:-0}" != "1" ]]; then
  echo "PIEVO_PREFILTER_INVALID_CANDIDATES must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${PIEVO_MAX_PREFILTER_REJECTIONS:-0}" =~ ^[0-9]+$ ]]; then
  echo "PIEVO_MAX_PREFILTER_REJECTIONS must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${PIEVO_MAX_PREFILTER_EMPTY_POOLS:-0}" =~ ^[0-9]+$ ]]; then
  echo "PIEVO_MAX_PREFILTER_EMPTY_POOLS must be a non-negative integer" >&2
  exit 2
fi

if [[ "$STAGE" == "formal" ]]; then
  TRACK="${PIEVO_RUN_TRACK:-matched_qwen235b}"
  if [[ "$TRACK" != "$MODEL_TRACK" ]] || [[ "$BUDGET" -ne 24 ]]; then
    echo "Formal runs require track=$MODEL_TRACK and exactly 24 primary evaluations" >&2
    exit 3
  fi
  if [[ "${PIEVO_CORRUPTION_MODE:-none}" != "none" ]]; then
    echo "Formal runs forbid diagnostic corruption" >&2
    exit 3
  fi
else
  TRACK="${PIEVO_RUN_TRACK:-${MODEL_TRACK}_reconstruction}"
  if [[ "$TRACK" != "${MODEL_TRACK}_reconstruction" ]] || [[ "$BUDGET" -lt 1 ]] || [[ "$BUDGET" -gt 8 ]]; then
    echo "Development runs require the $MODEL_TRACK reconstruction track and a 1-8 primary budget" >&2
    exit 3
  fi
fi

declare -A TASK_CONFIGS=(
  [NHO]="$ROOT/experiments/configs/official_pievo_nho_task.yaml"
  [MBO]="$ROOT/experiments/configs/official_pievo_mbo_task.yaml"
  [SPO]="$ROOT/experiments/configs/official_pievo_spo_task.yaml"
  [TMC]="$ROOT/experiments/configs/official_pievo_tmc_task.yaml"
)
declare -A MODEL_CONFIGS=(
  [NHO]="$ROOT/experiments/configs/official_pievo_nho_model.yaml"
  [MBO]="$ROOT/experiments/configs/official_pievo_mbo_model_short_smiles.yaml"
  [SPO]="$ROOT/experiments/configs/official_pievo_spo_model.yaml"
  [TMC]="$ROOT/experiments/configs/official_pievo_tmc_model.yaml"
)

if [[ "$METHOD" == "radp" ]]; then
  OFFICIAL_REPO="${PIEVO_RADP_REPO:-$ROOT/official_sources/PiEvo_radp_qwen235b}"
else
  OFFICIAL_REPO="${PIEVO_MATCHED_REPO:-$ROOT/official_sources/PiEvo_matched_qwen235b}"
fi
PYTHON_BIN="${PIEVO_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi
for required_path in "${TASK_CONFIGS[$TASK]}" "${MODEL_CONFIGS[$TASK]}" "$OFFICIAL_REPO/pievo/main.py"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing run input: $required_path" >&2
    exit 2
  fi
done
if [[ ! -f "$PROTOCOL" ]]; then
  echo "Missing protocol: $PROTOCOL" >&2
  exit 2
fi
for shared_path in \
  pievo/tools/candidate_validation.py \
  pievo/agents/experiment.py \
  pievo/word/instruction.py; do
  if ! cmp -s \
    "$ROOT/official_sources/PiEvo_matched_qwen235b/$shared_path" \
    "$ROOT/official_sources/PiEvo_radp_qwen235b/$shared_path"; then
    echo "Matched candidate-generation path differs between PiEvo and RADP: $shared_path" >&2
    exit 3
  fi
done

if [[ -z "${PIEVO_OUT:-}" ]]; then
  if [[ "$STAGE" == "formal" && "$MODEL_TRACK" == "matched_qwen235b" ]]; then
    OUT="$ROOT/experiments/results/${MODEL_TRACK}_${METHOD}_${TASK,,}_seed${SEED}_formal24"
  else
    OUT="$ROOT/experiments/results/development_${MODEL_TRACK}_${METHOD}_${TASK,,}_seed${SEED}_budget${BUDGET}"
  fi
else
  OUT="$PIEVO_OUT"
fi
if [[ "$OUT" != /* ]]; then
  out_parent="$(dirname "$OUT")"
  out_name="$(basename "$OUT")"
  mkdir -p "$out_parent"
  OUT="$(cd "$out_parent" && pwd)/$out_name"
fi
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Output directory is not fresh; refusing retry splicing: $OUT" >&2
  exit 3
fi
mkdir -p "$OUT"

export PIEVO_TASK="$TASK"
export PIEVO_METHOD="$METHOD"
export PIEVO_SEED="$SEED"
export PIEVO_RUN_STAGE="$STAGE"
export PIEVO_RUN_TRACK="$TRACK"
export PIEVO_MAX_PRIMARY_EVALS="$BUDGET"
export PIEVO_MAX_INVALID_ATTEMPTS="${PIEVO_MAX_INVALID_ATTEMPTS:-$([[ "$STAGE" == formal ]] && echo 8 || echo 3)}"
if [[ "$TASK" == "SPO" ]] || {
  [[ "$STAGE" == "formal" ]] && [[ "$TASK" == "MBO" || "$TASK" == "TMC" ]]
}; then
  # SPO, MBO, and TMC have deterministic released local validators. Keep the
  # shared prefilter explicit and matched before a paid evaluator action.
  export PIEVO_PREFILTER_INVALID_CANDIDATES="${PIEVO_PREFILTER_INVALID_CANDIDATES:-1}"
  export PIEVO_MAX_PREFILTER_REJECTIONS="${PIEVO_MAX_PREFILTER_REJECTIONS:-12}"
  export PIEVO_MAX_PREFILTER_EMPTY_POOLS="${PIEVO_MAX_PREFILTER_EMPTY_POOLS:-2}"
else
  export PIEVO_PREFILTER_INVALID_CANDIDATES="${PIEVO_PREFILTER_INVALID_CANDIDATES:-0}"
  export PIEVO_MAX_PREFILTER_REJECTIONS="${PIEVO_MAX_PREFILTER_REJECTIONS:-0}"
  export PIEVO_MAX_PREFILTER_EMPTY_POOLS="${PIEVO_MAX_PREFILTER_EMPTY_POOLS:-0}"
fi
export PIEVO_MAX_TOTAL_TOKENS="${PIEVO_MAX_TOTAL_TOKENS:-$([[ "$STAGE" == formal ]] && echo 350000 || echo 100000)}"
export PIEVO_MAX_RETRIES="${PIEVO_MAX_RETRIES:-0}"
export PIEVO_STOP_REQUESTED=0
export PIEVO_COMMON_GENERATOR="${PIEVO_COMMON_GENERATOR:-0}"
export PIEVO_AFE_AUTHORIZED_CANDIDATE=""
export PIEVO_AFE_DUPLICATE_RETRY_REQUESTED=0
export PYTHONHASHSEED="$SEED"
export WORKING_RESULT_DIR="$OUT"
export PIEVO_DISABLE_PLOTS="${PIEVO_DISABLE_PLOTS:-1}"
export PIEVO_EMBEDDING_OFFLINE="${PIEVO_EMBEDDING_OFFLINE:-1}"
EMBEDDING_MODEL_DEFAULT="/Users/michael/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export PIEVO_EMBEDDING_MODEL_PATH="${PIEVO_EMBEDDING_MODEL_PATH:-$EMBEDDING_MODEL_DEFAULT}"
export PIEVO_SURROGATE_ROOT="${PIEVO_SURROGATE_ROOT:-$OFFICIAL_REPO/surrogates/envs}"
export PIEVO_CHEMBL_MODEL_PATH="${PIEVO_CHEMBL_MODEL_PATH:-$OFFICIAL_REPO/surrogates/envs/ChEMBL35/models/best_r2_model.pt}"
if [[ "$TASK" == "MBO" ]]; then
  # This molecule was already measured in retained development evidence. It
  # may inform a declared structural prior but cannot be counted again as a
  # fresh benchmark primary, including through an equivalent SMILES spelling.
  export PIEVO_MBO_FORBIDDEN_CANDIDATES="${PIEVO_MBO_FORBIDDEN_CANDIDATES:-C1CCCCCCC2C1CCCCCCC3C2CCCCCCC3}"
  # AFE reads only canonical_smiles from this retained reconstruction.  The
  # activity columns remain in the artifact for provenance but never enter the
  # candidate selector.
  export PIEVO_AFE_POOL_PATH="${PIEVO_AFE_POOL_PATH:-$ROOT/experiments/results/official_chembl35_pool_seed0/chembl35_pool_seed0.jsonl}"
  export PIEVO_AFE_POOL_LIMIT="${PIEVO_AFE_POOL_LIMIT:-50000}"
  export PIEVO_AFE_POOL_PROPOSALS="${PIEVO_AFE_POOL_PROPOSALS:-256}"
elif [[ "$TASK" == "SPO" ]]; then
  # The released PiFlow data contributes only its public formula column to
  # the SPO AFE stream. The loader never reads tc/utc/journal observations.
  export PIEVO_AFE_POOL_PATH="${PIEVO_AFE_POOL_PATH:-$ROOT/official_sources/PiFlow/envs/AgenX_Supercon/data/superconductor_data.tsv}"
  export PIEVO_AFE_POOL_LIMIT="${PIEVO_AFE_POOL_LIMIT:-50000}"
  export PIEVO_AFE_POOL_PROPOSALS="${PIEVO_AFE_POOL_PROPOSALS:-256}"
else
  export PIEVO_MBO_FORBIDDEN_CANDIDATES=""
  export PIEVO_AFE_POOL_PATH=""
  export PIEVO_AFE_POOL_LIMIT=0
  export PIEVO_AFE_POOL_PROPOSALS=0
fi
export PIEVO_OFFICIAL_PIFLOW_REPO="${PIEVO_OFFICIAL_PIFLOW_REPO:-$ROOT/official_sources/PiFlow}"
export PIEVO_EXPERIMENT_ROOT="$ROOT"
export PYTHONPATH="$OFFICIAL_REPO/surrogates/envs:$OFFICIAL_REPO:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$METHOD" == "radp" ]]; then
  if [[ "$TASK" == "SPO" ]]; then
    export PIEVO_ALGORITHM_ID=radp_search_afe_spo_v3
    export PIEVO_RADP_SEARCH_STRATEGY="${PIEVO_RADP_SEARCH_STRATEGY:-afe_spo_v2}"
    export PIEVO_AFE_SCHEMA_VERSION="${PIEVO_AFE_SCHEMA_VERSION:-spo_v2}"
  else
    export PIEVO_ALGORITHM_ID=radp_search_afe_v9
    # AFE v9 is the performance track. It expands the structural candidate pool
    # while keeping source-gate and principle evidence out of acquisition.
    export PIEVO_RADP_SEARCH_STRATEGY="${PIEVO_RADP_SEARCH_STRATEGY:-afe_v9}"
    export PIEVO_AFE_SCHEMA_VERSION="${PIEVO_AFE_SCHEMA_VERSION:-v9}"
  fi
  export PIEVO_RADP_WIDE_SEED_BANK="${PIEVO_RADP_WIDE_SEED_BANK:-0}"
  export PIEVO_AFE_PORTFOLIO_SIZE="${PIEVO_AFE_PORTFOLIO_SIZE:-18}"
  export PIEVO_SOURCE_GATE=1
  export PIEVO_SOURCE_GATE_MODE=active
  export PIEVO_SOURCE_GATE_PROVIDER="official_${TASK,,}"
  export PIEVO_REVISION_PROTOCOL=1
  export PIEVO_ACTIVE_BACKTRACK=1
  # Unaccepted RADP principles remain in internal lifecycle/GP state but are
  # not allowed to steer new hypothesis generation.  This is the controlled
  # candidate-quality diagnostic; it is recorded in the manifest.
  export PIEVO_ACCEPTED_PRINCIPLE_ONLY=1
  export PIEVO_BOOTSTRAP_PRIOR_CONTEXT="${PIEVO_BOOTSTRAP_PRIOR_CONTEXT:-1}"
  # Raw group messages remain available to the controller and audit layer,
  # but role LLMs see only the current provenance-filtered guidance.
  export PIEVO_GUIDANCE_ONLY_CONTEXT=1
  # Allow clean, reconciled reward history to guide neutral candidate search
  # without exposing unaccepted principles or verifier/source-gate details.
  if [[ "$TASK" == "MBO" ]]; then
    export PIEVO_NEUTRAL_REWARD_GUIDED="${PIEVO_NEUTRAL_REWARD_GUIDED:-1}"
  else
    export PIEVO_NEUTRAL_REWARD_GUIDED=0
  fi
  export PIEVO_SOURCE_GATE_MAX_CALLS="${PIEVO_SOURCE_GATE_MAX_CALLS:-8}"
  export PIEVO_SOURCE_GATE_TOLERANCE="${PIEVO_SOURCE_GATE_TOLERANCE:-1e-6}"
  # The released surrogate repeat is a diagnostic only.  A real independent
  # verifier must be explicitly configured after its own readiness gate.
  export PIEVO_MBO_INDEPENDENT_VERIFIER="${PIEVO_MBO_INDEPENDENT_VERIFIER:-}"
else
  export PIEVO_ALGORITHM_ID=pievo_matched_release_v1
  export PIEVO_RADP_SEARCH_STRATEGY=legacy_v1
  export PIEVO_RADP_WIDE_SEED_BANK=0
  export PIEVO_SOURCE_GATE=0
  export PIEVO_SOURCE_GATE_MODE=off
  export PIEVO_SOURCE_GATE_PROVIDER=none
  export PIEVO_REVISION_PROTOCOL=0
  export PIEVO_ACTIVE_BACKTRACK=0
  export PIEVO_ACCEPTED_PRINCIPLE_ONLY=0
  export PIEVO_GUIDANCE_ONLY_CONTEXT=0
  export PIEVO_NEUTRAL_REWARD_GUIDED=0
  export PIEVO_SOURCE_GATE_MAX_CALLS=0
  export PIEVO_MBO_INDEPENDENT_VERIFIER=""
fi

# Development-only AFE/source-gate factor audit.  The AFE implementation is
# held fixed while the source gate is disabled, shadowed, or made active.
# This branch is opt-in and cannot affect formal runs.
if [[ "$STAGE" == "development" && "${PIEVO_EXPLORATORY_AFE_GATE_MODES:-0}" == "1" && "$METHOD" == "radp" ]]; then
  case "${PIEVO_EXPLORATORY_GATE_MODE:-active}" in
    off)
      export PIEVO_SOURCE_GATE=0
      export PIEVO_SOURCE_GATE_MODE=off
      # Keep the evaluator provider fixed; this factor disables only source
      # gate authorization.
      export PIEVO_SOURCE_GATE_PROVIDER="official_${TASK,,}"
      export PIEVO_SOURCE_GATE_BEHAVIOR=off
      ;;
    shadow)
      export PIEVO_SOURCE_GATE=1
      export PIEVO_SOURCE_GATE_MODE=shadow
      export PIEVO_SOURCE_GATE_PROVIDER="official_${TASK,,}"
      export PIEVO_SOURCE_GATE_BEHAVIOR=shadow
      ;;
    active)
      export PIEVO_SOURCE_GATE=1
      export PIEVO_SOURCE_GATE_MODE=active
      export PIEVO_SOURCE_GATE_PROVIDER="official_${TASK,,}"
      export PIEVO_SOURCE_GATE_BEHAVIOR=fixed
      ;;
    *)
      echo "PIEVO_EXPLORATORY_GATE_MODE must be off, shadow, or active" >&2
      exit 3
      ;;
  esac
  export PIEVO_AFE_FORCE_ENABLED=1
fi

if [[ "$STAGE" == "formal" && "$PIEVO_RADP_SEARCH_STRATEGY" == "portfolio_v2" ]]; then
  echo "portfolio_v2 is retired; use the frozen AFE strategy for the RADP performance track" >&2
  exit 3
fi
if [[ "$STAGE" == "formal" && "$PIEVO_COMMON_GENERATOR" != "0" ]]; then
  echo "common-generator controls are development-only; formal treatment uses the frozen AFE strategy" >&2
  exit 3
fi

ATTESTATION="$OUT/runtime_attestation.json"
preflight_args=(
  --task "$TASK"
  --method "$METHOD"
  --seed "$SEED"
  --repository "$OFFICIAL_REPO"
  --task-config "${TASK_CONFIGS[$TASK]}"
  --model-config "${MODEL_CONFIGS[$TASK]}"
  --protocol "$PROTOCOL"
  --recovery "$ROOT/experiments/outcome_blind_candidate_recovery.py"
  --output "$ATTESTATION"
)
if [[ "$STAGE" == "development" ]]; then
  preflight_args+=(--development)
fi
env -u ZJU_TASK_API_KEY -u OPENAI_API_KEY \
  "$PYTHON_BIN" "$ROOT/experiments/formal_runtime_preflight.py" "${preflight_args[@]}"

if [[ "$MODEL_TRACK" == "gpt56luna" ]]; then
  APPROVAL="${PIEVO_FORMAL_APPROVAL:-$ROOT/experiments/results/formal_approvals/${MODEL_TRACK}_${METHOD}_${TASK,,}_v2.json}"
else
  APPROVAL="${PIEVO_FORMAL_APPROVAL:-$ROOT/experiments/results/formal_approvals/${MODEL_TRACK}_${METHOD}_${TASK,,}_v1.json}"
fi
if [[ "$STAGE" == "formal" && "$MODEL_TRACK" == "matched_qwen235b" ]]; then
  PREFLIGHT="${PIEVO_FORMAL_PREFLIGHT:-$ROOT/experiments/results/official_formal_preflight_afe_v11.json}"
  FREEZE="${PIEVO_PROTOCOL_FREEZE:-$ROOT/experiments/results/matched_qwen235b_protocol_freeze_afe_v11.json}"
  AFE_PREFLIGHT="${PIEVO_AFE_PREFLIGHT:-$ROOT/experiments/results/afe_zero_api_preflight_v11.json}"
  if ! "$PYTHON_BIN" - "$PREFLIGHT" "$FREEZE" "$PROTOCOL" "$ROOT" "$AFE_PREFLIGHT" "$ROOT/experiments/run_matched_qwen235b_task.sh" "$ATTESTATION" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

preflight_path, freeze_path, protocol_path, root, afe_preflight_path, runner_path, attestation_path = map(Path, sys.argv[1:])
try:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    afe_preflight = json.loads(afe_preflight_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
core_paths = {
    "pievo_main": root / "official_sources/PiEvo_matched_qwen235b/pievo/main.py",
    "radp_main": root / "official_sources/PiEvo_radp_qwen235b/pievo/main.py",
    "pievo_evolveflow": root / "official_sources/PiEvo_matched_qwen235b/pievo/group/evolveflow.py",
    "radp_evolveflow": root / "official_sources/PiEvo_radp_qwen235b/pievo/group/evolveflow.py",
    "pievo_manage": root / "official_sources/PiEvo_matched_qwen235b/pievo/group/manage.py",
    "radp_manage": root / "official_sources/PiEvo_radp_qwen235b/pievo/group/manage.py",
    "pievo_experiment_agent": root / "official_sources/PiEvo_matched_qwen235b/pievo/agents/experiment.py",
    "radp_experiment_agent": root / "official_sources/PiEvo_radp_qwen235b/pievo/agents/experiment.py",
    "pievo_instruction": root / "official_sources/PiEvo_matched_qwen235b/pievo/word/instruction.py",
    "radp_instruction": root / "official_sources/PiEvo_radp_qwen235b/pievo/word/instruction.py",
    "pievo_candidate_validation": root / "official_sources/PiEvo_matched_qwen235b/pievo/tools/candidate_validation.py",
    "radp_candidate_validation": root / "official_sources/PiEvo_radp_qwen235b/pievo/tools/candidate_validation.py",
    "pievo_chembl_tool": root / "official_sources/PiEvo_matched_qwen235b/pievo/tools/tool_chembl_official.py",
    "radp_chembl_tool": root / "official_sources/PiEvo_radp_qwen235b/pievo/tools/tool_chembl_official.py",
    "pievo_model_config": root / "official_sources/PiEvo_matched_qwen235b/config/model.yaml",
    "radp_model_config": root / "official_sources/PiEvo_radp_qwen235b/config/model.yaml",
    "pievo_bayesian": root / "official_sources/PiEvo_matched_qwen235b/pievo/word/bayesian.py",
    "radp_bayesian": root / "official_sources/PiEvo_radp_qwen235b/pievo/word/bayesian.py",
    "radp_afe": root / "official_sources/PiEvo_radp_qwen235b/pievo/radp/afe.py",
    "radp_wide_candidate": root / "official_sources/PiEvo_radp_qwen235b/pievo/radp/wide_candidate.py",
  "radp_capabilities": root / "official_sources/PiEvo_radp_qwen235b/pievo/radp/capabilities.json",
  "pievo_embedding": root / "official_sources/PiEvo_matched_qwen235b/pievo/word/embedding.py",
  "radp_embedding": root / "official_sources/PiEvo_radp_qwen235b/pievo/word/embedding.py",
  "common_candidate_generator": root / "experiments/common_candidate_generator.py",
}
if not all(path.is_file() for path in core_paths.values()):
    raise SystemExit(1)
hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in core_paths.items()}
protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.is_file() else None
runner_hash = hashlib.sha256(runner_path.read_bytes()).hexdigest() if runner_path.is_file() else None
raise SystemExit(0 if (
    preflight.get("formal_matched_run_ready") is True
    and afe_preflight.get("passed") is True
    and afe_preflight.get("api_calls") == 0
    and afe_preflight.get("protocol_sha256") == protocol_hash
    and freeze.get("protocol_sha256") == protocol_hash
    and freeze.get("runner_sha256") == runner_hash
    and freeze.get("worktree_core_hashes") == hashes
    and freeze.get("embedding_model") == attestation.get("embedding_model")
) else 1)
PY
  then
    echo "G0 readiness or the frozen code/protocol hash is stale" >&2
    exit 3
  fi
  if ! PYTHONPATH="$ROOT/experiments" "$PYTHON_BIN" - "$APPROVAL" "$ATTESTATION" "$METHOD" "$TASK" <<'PY'
import json
import sys
from pathlib import Path
from formal_runtime_contract import formal_approval_is_valid

approval_path, attestation_path = map(Path, sys.argv[1:3])
method, task = sys.argv[3:5]
try:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if formal_approval_is_valid(
    approval,
    method=method,
    task=task,
    code_hashes=attestation.get("code_hashes", {}),
    embedding_model=attestation.get("embedding_model", {}),
) else 1)
PY
  then
    echo "Missing or stale bounded-development approval: $APPROVAL" >&2
    exit 3
  fi
  export PIEVO_MATCHED_EVALUATOR_ASSETS_VERIFIED=1
elif [[ "$STAGE" == "formal" ]]; then
  # New model tracks reuse the runtime contract but must have their own
  # bounded-development approval.  The Qwen v11 freeze is intentionally not
  # reused because the model, decode protocol, and client provenance differ.
  FREEZE="${PIEVO_MODEL_TRACK_FREEZE:-$ROOT/experiments/results/${MODEL_TRACK}_${METHOD}_${TASK,,}_protocol_freeze_v2.json}"
  if ! "$PYTHON_BIN" - "$APPROVAL" "$FREEZE" "$MODEL" "$MODEL_TRACK" "$PROTOCOL" "$ROOT/experiments/run_matched_qwen235b_task.sh" "$ATTESTATION" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

approval_path, freeze_path, model, model_track, protocol_path, runner_path, attestation_path = map(Path, sys.argv[1:])
try:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
if approval.get("passed") is not True or approval.get("model") != str(model) or approval.get("model_track") != str(model_track):
    raise SystemExit(1)
protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
runner_hash = hashlib.sha256(runner_path.read_bytes()).hexdigest()
if approval.get("protocol_sha256") != protocol_hash:
    raise SystemExit(1)
if (
    freeze.get("passed") is not True
    or freeze.get("model") != str(model)
    or freeze.get("model_track") != str(model_track)
    or freeze.get("protocol_sha256") != protocol_hash
    or freeze.get("runner_sha256") != runner_hash
    or freeze.get("code_hashes") != attestation.get("code_hashes")
    or freeze.get("embedding_model") != attestation.get("embedding_model")
):
    raise SystemExit(1)
PY
  then
    echo "Missing or stale model-track development approval: $APPROVAL" >&2
    exit 3
  fi
  export PIEVO_MATCHED_EVALUATOR_ASSETS_VERIFIED=1
else
  APPROVAL=""
  export PIEVO_MATCHED_EVALUATOR_ASSETS_VERIFIED=0
fi
export PIEVO_OFFICIAL_ASSETS_VERIFIED=0

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Offline dry-run passed; no API key was read and no model request was made."
  exit 0
fi

if [[ -z "${ZJU_TASK_API_KEY:-}" ]]; then
  key_file="${ZJU_TASK_API_KEY_FILE:-/private/tmp/zju_task_api_key}"
  if [[ -r "$key_file" ]]; then
    ZJU_TASK_API_KEY="$(<"$key_file")"
    export ZJU_TASK_API_KEY
  fi
fi
if [[ -z "${ZJU_TASK_API_KEY:-}" ]] && command -v security >/dev/null 2>&1; then
  ZJU_TASK_API_KEY="$(security find-generic-password \
    -a "${ZJU_TASK_KEYCHAIN_ACCOUNT:-ai4ai}" \
    -s "${ZJU_TASK_KEYCHAIN_SERVICE:-zjuapi.com}" -w 2>/dev/null || true)"
  export ZJU_TASK_API_KEY
fi
if [[ -z "${ZJU_TASK_API_KEY:-}" ]]; then
  echo "Set ZJU_TASK_API_KEY or ZJU_TASK_API_KEY_FILE after all offline gates pass" >&2
  exit 2
fi
export UTIL_LLM_CONFIG_API_KEY="$ZJU_TASK_API_KEY"
export UTIL_LLM_CONFIG_NAME="$MODEL"
export UTIL_LLM_CONFIG_BASE_URL="${PIEVO_AGENT_BASE_URL:-https://zjuapi.com/v1}"
export PIEVO_AGENT_MODEL="$MODEL"
export PIEVO_MODEL_TRACK="$MODEL_TRACK"
export PIEVO_UTIL_MODEL_FORCE="$MODEL"
export PIEVO_UTIL_BASE_URL_FORCE="${PIEVO_AGENT_BASE_URL:-https://zjuapi.com/v1}"
if [[ -n "$REASONING_EFFORT" ]]; then
  export PIEVO_REASONING_EFFORT="$REASONING_EFFORT"
fi
export PIEVO_DECODE_PROTOCOL="${PIEVO_DECODE_PROTOCOL:-$ROOT/experiments/configs/controlled_qwen235b_protocol_v1.json}"
if [[ ! -f "$PIEVO_DECODE_PROTOCOL" ]]; then
  echo "Missing controlled decode protocol: $PIEVO_DECODE_PROTOCOL" >&2
  exit 2
fi
if [[ "${PIEVO_AGENT_TEMPERATURE:-0.6}" != "0.6" ]]; then
  echo "Controlled reruns require PIEVO_AGENT_TEMPERATURE=0.6" >&2
  exit 3
fi
if [[ "${PIEVO_AGENT_MAX_TOKENS:-4096}" != "4096" ]]; then
  echo "Controlled reruns require PIEVO_AGENT_MAX_TOKENS=4096" >&2
  exit 3
fi

manifest_args=(
  --output "$OUT"
  --task "$TASK"
  --method "$METHOD"
  --seed "$SEED"
  --budget "$BUDGET"
  --official-repo "$OFFICIAL_REPO"
  --task-config "${TASK_CONFIGS[$TASK]}"
  --model-config "${MODEL_CONFIGS[$TASK]}"
  --protocol "$PROTOCOL"
  --runtime-attestation "$ATTESTATION"
  --stage "$STAGE"
  --token-ceiling "$PIEVO_MAX_TOTAL_TOKENS"
  --invalid-limit "$PIEVO_MAX_INVALID_ATTEMPTS"
)
if [[ -n "$APPROVAL" ]]; then
  manifest_args+=(--formal-approval "$APPROVAL")
fi
"$PYTHON_BIN" "$ROOT/experiments/write_run_manifest.py" "${manifest_args[@]}" --status started

cd "$OFFICIAL_REPO"
agent_model_args=(
  --agent_base_url_force "${PIEVO_AGENT_BASE_URL:-https://zjuapi.com/v1}"
  --agent_model_name_force "${PIEVO_AGENT_MODEL:-qwen3-235b-a22b-instruct-2507}"
  --agent_temperature_force "0.6"
  --agent_max_tokens_force "4096"
)
if [[ -n "${PIEVO_REASONING_EFFORT:-}" ]]; then
  agent_model_args+=(--agent_reasoning_effort_force "$PIEVO_REASONING_EFFORT")
fi
set +e
"$PYTHON_BIN" -m pievo.main \
  --task_config "${TASK_CONFIGS[$TASK]}" \
  --model_config "${MODEL_CONFIGS[$TASK]}" \
  --output_dir "$OUT" \
  --max_turn "${PIEVO_MAX_TURN:-280}" \
  --warm_up_rounds "${PIEVO_WARMUP_ROUNDS:-5}" \
  --anomaly_threshold "${PIEVO_ANOMALY_THRESHOLD:-0.85}" \
  --anomaly_cnt_threshold "${PIEVO_ANOMALY_COUNT_THRESHOLD:-3}" \
  --budget "${PIEVO_GROUP_BUDGET:-$((BUDGET + 16))}" \
  "${agent_model_args[@]}"
run_status=$?
set -e

# The runtime resolves ${ZJU_TASK_API_KEY} in copied config snapshots. Those
# copies are useful provenance, but credentials are not research evidence.
# Redact only the supplied credential after each run, including failures.
if [[ -n "${ZJU_TASK_API_KEY:-}" ]]; then
  while IFS= read -r -d '' copied_file; do
    PIEVO_REDACT_VALUE="$ZJU_TASK_API_KEY" perl -0pi -e 's/\Q$ENV{PIEVO_REDACT_VALUE}\E/\$\{ZJU_TASK_API_KEY\}/g' "$copied_file"
  done < <(rg -l -0 -F -- "$ZJU_TASK_API_KEY" "$OUT" 2>/dev/null || true)
fi

if [[ "$run_status" -eq 0 ]]; then
  audit_count="$($PYTHON_BIN - "$OUT" "$TASK" "$BUDGET" <<'PY'
import json
import math
import sys
from pathlib import Path

out, task, budget = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
names = {"NHO": "official_nho_tool_audit.jsonl", "MBO": "official_chembl_tool_audit.jsonl", "SPO": "official_spo_tool_audit.jsonl", "TMC": "official_tmc_tool_audit.jsonl"}
count = 0
path = out / names[task]
if path.is_file():
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        output = row.get("output")
        if (
            str(row.get("tool_name", "")).startswith("evaluate_")
            and row.get("success") is True
            and row.get("duplicate_suppressed") is not True
            and row.get("record_type") not in {"duplicate_suppressed", "verification"}
            and isinstance(output, (int, float))
            and not isinstance(output, bool)
            and math.isfinite(float(output))
        ):
            count += 1
print(count)
PY
  )"
  if [[ "$audit_count" -ne "$BUDGET" ]]; then
    echo "Run ended without exactly $BUDGET valid primary evaluations; found $audit_count" >&2
    run_status=10
  fi
fi

if [[ "$run_status" -eq 0 ]]; then
  if ! PYTHONPATH="$ROOT/experiments" "$PYTHON_BIN" - \
    "$OUT" "$SEED" "$PIEVO_ALGORITHM_ID" "$PIEVO_MAX_TOTAL_TOKENS" "$METHOD" "$TASK" <<'PY'
import json
import os
import sys
from pathlib import Path

from formal_usage import read_cumulative_usage

out = Path(sys.argv[1])
seed = int(sys.argv[2])
algorithm_id = sys.argv[3]
ceiling = int(sys.argv[4])
method = sys.argv[5]
task = sys.argv[6]
usage = read_cumulative_usage(out / "token_usage.json")
try:
    seed_proof = json.loads((out / "seed_application.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    seed_proof = {}
clients = seed_proof.get("llm_client_seeds")
seed_ok = bool(
    seed_proof.get("schema_version") == "pievo_seed_application.v1"
    and seed_proof.get("seed") == seed
    and seed_proof.get("algorithm_id") == algorithm_id
    and seed_proof.get("seed_applied") is True
    and isinstance(clients, list)
    and clients
    and all(value == seed for value in clients)
)
token_ok = bool(
    usage["snapshot_index"] is not None
    and 0 < int(usage["total"] or 0) <= ceiling
)
radp_ok = method != "radp"
if method == "radp":
    audit_path = out / "radp_runtime_audit.jsonl"
    expected_provider = f"official_{task.lower()}"
    expected_mode = "active"
    expected_enabled = True
    exploratory = (
        os.environ.get("PIEVO_RUN_STAGE", "formal") == "development"
        and os.environ.get("PIEVO_EXPLORATORY_AFE_GATE_MODES", "0") == "1"
    )
    if exploratory:
        expected_mode = os.environ.get("PIEVO_EXPLORATORY_GATE_MODE", "active")
        expected_enabled = expected_mode != "off"
    if audit_path.is_file():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (
                row.get("event") == "runtime_start"
                and row.get("algorithm_id") == algorithm_id
                and row.get("mode") == expected_mode
                and row.get("provider") == expected_provider
                and row.get("source_gate_enabled") is expected_enabled
                and row.get("revision_protocol_enabled") is True
                and row.get("backtrack_enabled") is True
            ):
                radp_ok = True
                break
raise SystemExit(0 if seed_ok and token_ok and radp_ok else 1)
PY
  then
    echo "Run lacks valid seed, token-ceiling, or RADP runtime evidence; marking failed" >&2
    run_status=11
  fi
fi

"$PYTHON_BIN" "$ROOT/experiments/write_run_manifest.py" "${manifest_args[@]}" \
  --status "$([[ "$run_status" -eq 0 ]] && echo completed || echo failed)" \
  --exit-code "$run_status"
exit "$run_status"
