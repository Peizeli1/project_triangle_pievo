#!/usr/bin/env bash
set -euo pipefail

# A bounded, paired MBO development screen for the frozen RADP-Search-AFE
# implementation.  This is deliberately separate from the formal runner: it
# checks live integration on one official evaluator, but cannot unlock the
# official matrix or an external benchmark claim.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PIEVO_PYTHON:-$ROOT/.venv/bin/python}"
BASE_PROTOCOL="$ROOT/experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
AFE_PROTOCOL="$ROOT/experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json"
PROTOCOL="${PIEVO_PROTOCOL_PATH:-$AFE_PROTOCOL}"
SEED="${PIEVO_SEED:-0}"
BUDGET="${PIEVO_BUDGET:-6}"
DRY_RUN="${PIEVO_DRY_RUN:-1}"
TOKEN_CEILING="${PIEVO_MAX_TOTAL_TOKENS:-75000}"
AGENT_MAX_TOKENS="${PIEVO_AGENT_MAX_TOKENS:-2048}"
SCREEN_MODE="${PIEVO_MINIMAL_MODE:-clean}"
PREFILTER_INVALID_CANDIDATES=1
MAX_PREFILTER_REJECTIONS=12
MAX_PREFILTER_EMPTY_POOLS=2
RADP_SEARCH_STRATEGY="${PIEVO_RADP_SEARCH_STRATEGY:-afe_v8}"
# Keep the development comparison live even when the baseline LLM emits an
# invalid batch.  In the default ``auto`` mode PiEvo receives the common
# outcome-blind stream and RADP owns the AFE structural stream.  An explicit
# 0/1 value remains available for a common-generator control.
COMMON_GENERATOR_MODE="${PIEVO_COMMON_GENERATOR:-auto}"
# The retired fixed wide seed-bank injection belongs to the historical
# portfolio diagnostic.  AFE gets its candidates from the structural pool and
# its deterministic acquisition portfolio instead.
RADP_WIDE_SEED_BANK="${PIEVO_RADP_WIDE_SEED_BANK:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[01]$ ]]; then
  echo "PIEVO_SEED must be 0 or 1 for the paired screen" >&2
  exit 2
fi
if [[ "$BUDGET" -ne 6 ]]; then
  echo "This screen is fixed at exactly 6 primary evaluations per method" >&2
  exit 3
fi
if [[ "$SCREEN_MODE" != "clean" && "$SCREEN_MODE" != "source_confusion" ]]; then
  echo "PIEVO_MINIMAL_MODE must be clean or source_confusion" >&2
  exit 3
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "PIEVO_DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "$RADP_SEARCH_STRATEGY" != "afe_v8" ]]; then
  echo "This development screen is frozen to RADP strategy afe_v8" >&2
  exit 3
fi
if [[ "$RADP_WIDE_SEED_BANK" != "0" ]]; then
  echo "AFE development runs do not enable the retired wide seed-bank injection" >&2
  exit 3
fi
if [[ "$COMMON_GENERATOR_MODE" != "auto" && "$COMMON_GENERATOR_MODE" != "0" && "$COMMON_GENERATOR_MODE" != "1" ]]; then
  echo "PIEVO_COMMON_GENERATOR must be auto, 0, or 1" >&2
  exit 2
fi
if [[ "$PROTOCOL" != "$AFE_PROTOCOL" ]]; then
  echo "This development screen is frozen to the AFE v8 protocol" >&2
  exit 3
fi
if [[ "$DRY_RUN" == "0" && "${PIEVO_MINIMAL_SCREEN_APPROVAL:-}" != "YES" ]]; then
  echo "Set PIEVO_MINIMAL_SCREEN_APPROVAL=YES for the paid 12-primary screen" >&2
  exit 3
fi

# The default dry-run location is outside the project.  A real run must name
# its output root explicitly so an accidental invocation cannot overwrite an
# existing result or silently splice a retry into an earlier run.
if [[ -n "${PIEVO_MINIMAL_OUT_ROOT:-}" ]]; then
  OUT_ROOT="$PIEVO_MINIMAL_OUT_ROOT"
else
  if [[ "$DRY_RUN" == "1" ]]; then
    OUT_ROOT="$(mktemp -d "/private/tmp/ai4ai_official_mbo_screen_seed${SEED}.XXXXXX")"
  else
    echo "PIEVO_MINIMAL_OUT_ROOT is required for a real screen" >&2
    exit 3
  fi
fi
if [[ "$OUT_ROOT" != /* || "$OUT_ROOT" == "/" || "$OUT_ROOT" == "$ROOT" ]]; then
  echo "PIEVO_MINIMAL_OUT_ROOT must be an absolute, specific output directory" >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"

if [[ "$SCREEN_MODE" == "source_confusion" ]]; then
  CORRUPTION_MODE=one_shot_spike
  CORRUPTION_CALLS=2,3
  CORRUPTION_OFFSET=2.0
  # This is a preregistered development trigger. It guarantees that the
  # injected event reaches RADP's verifier within a six-primary screen.
  SOURCE_GATE_SCORE_GE=0.0
else
  CORRUPTION_MODE=none
  CORRUPTION_CALLS=""
  CORRUPTION_OFFSET=0.0
  SOURCE_GATE_SCORE_GE=""
fi

# Keep the executable diagnostic schedule bound to the frozen protocol.  This
# check is deliberately before either method starts, so a stale runner cannot
# spend API budget under a different corruption schedule or gate threshold.
if [[ "$SCREEN_MODE" == "source_confusion" ]]; then
  "$PYTHON_BIN" - \
    "$BASE_PROTOCOL" \
    "$CORRUPTION_MODE" "$CORRUPTION_CALLS" "$CORRUPTION_OFFSET" "$SOURCE_GATE_SCORE_GE" \
    "$PREFILTER_INVALID_CANDIDATES" "$MAX_PREFILTER_REJECTIONS" \
    "$MAX_PREFILTER_EMPTY_POOLS" <<'PY'
import json
import sys
from pathlib import Path

path, mode, calls, offset, score_ge, prefilter, rejection_limit, empty_limit = sys.argv[1:]
config = json.loads(Path(path).read_text(encoding="utf-8"))
schedule = config["minimal_controlled_api_screen"]["corruption_schedule"]
prefilter_config = config["minimal_controlled_api_screen"]["candidate_prefilter"]
expected_calls = ",".join(str(value) for value in schedule["calls"])
if (
    mode != schedule["mode"]
    or calls != expected_calls
    or float(offset) != float(schedule["offset"])
    or float(score_ge) != float(schedule["source_gate_score_ge"])
    or (prefilter == "1") is not bool(prefilter_config["enabled"])
    or int(rejection_limit) != int(prefilter_config["max_unique_rejections"])
    or int(empty_limit) != int(prefilter_config["max_consecutive_empty_pools"])
):
    raise SystemExit(
        "minimal source-confusion schedule does not match the frozen diagnostic protocol"
    )
PY
fi

# Confirm the zero-API mechanism gate before either paired preflight.  Exit
# status alone is insufficient: the artifact must explicitly authorize this
# bounded controlled-development runner (never the formal/broad matrix).
OFFLINE_GATE_ARTIFACT="$OUT_ROOT/minimal_radp_offline_gate.json"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
  "$ROOT/experiments/minimal_radp_offline_gate.py" --seeds 32 \
  --output "$OFFLINE_GATE_ARTIFACT" \
  >/dev/null
"$PYTHON_BIN" - "$OFFLINE_GATE_ARTIFACT" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("api_calls") != 0:
    raise SystemExit("offline gate unexpectedly records API calls")
if value.get("controlled_development_screen_ready") is not True:
    raise SystemExit("controlled MBO development screen is not offline-ready")
if value.get("broad_api_matrix_unlocked") is not False:
    raise SystemExit("offline gate must not unlock the broad API matrix")
PY

# Reconstruct the active v8 outcome-blind structural stream before any
# credential is consumed.  A stale protocol, source reader, or candidate
# boundary stops the screen.
AFE_STREAM_AUDIT="$OUT_ROOT/radp_afe_v8_preflight.json"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
  "$ROOT/experiments/audit_afe_preflight.py" \
  --output "$AFE_STREAM_AUDIT" \
  >/dev/null
"$PYTHON_BIN" - "$AFE_STREAM_AUDIT" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("api_calls") != 0 or value.get("passed") is not True:
    raise SystemExit("AFE v8 zero-API preflight did not pass")
if value.get("schema_version") != "radp_search_afe_zero_api_preflight.v8":
    raise SystemExit("unexpected AFE v8 preflight schema")
PY

# The wide-search surrogate audit is useful diagnostics, but it is not a
# prerequisite for the paired live screen.  Requiring a long local sweep here
# delayed the only result that can answer the SOTA question.  The live screen
# still enforces the shared validator, frozen AFE stream, fresh outputs, exact
# primary budget, and its complete audit below.

# Check credential availability only after every zero-API gate has passed and
# before creating a method output directory. This keeps a missing credential
# from leaving a misleading partial run behind.
if [[ "$DRY_RUN" == "0" && -z "${ZJU_TASK_API_KEY:-}" ]]; then
  key_file="${ZJU_TASK_API_KEY_FILE:-/private/tmp/zju_task_api_key}"
  if [[ ! -r "$key_file" || ! -s "$key_file" ]]; then
    echo "Set ZJU_TASK_API_KEY or ZJU_TASK_API_KEY_FILE after all offline gates pass" >&2
    exit 2
  fi
fi

if [[ "$SEED" == "0" ]]; then
  METHODS=(pievo radp)
else
  METHODS=(radp pievo)
fi

METHOD_FAILURE_STATUS=0
METHOD_FAILURE_NAME=""
for method in "${METHODS[@]}"; do
  if [[ "$COMMON_GENERATOR_MODE" == "auto" ]]; then
    if [[ "$method" == "pievo" ]]; then
      method_common_generator=1
    else
      method_common_generator=0
    fi
  else
    method_common_generator="$COMMON_GENERATOR_MODE"
  fi
  output="$OUT_ROOT/$method"
  if [[ -e "$output" ]] && find "$output" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Output is not fresh; refusing retry splicing: $output" >&2
    exit 3
  fi
  runner_env=(
    PIEVO_TASK=MBO \
    PIEVO_METHOD="$method" \
    PIEVO_SEED="$SEED" \
    PIEVO_BUDGET="$BUDGET" \
    PIEVO_RUN_STAGE=development \
    PIEVO_RUN_TRACK=matched_qwen235b_reconstruction \
    PIEVO_PROTOCOL_PATH="$PROTOCOL" \
    PIEVO_DRY_RUN="$DRY_RUN" \
    PIEVO_OUT="$output" \
    PIEVO_MAX_TOTAL_TOKENS="$TOKEN_CEILING" \
    PIEVO_AGENT_MAX_TOKENS="$AGENT_MAX_TOKENS" \
    PIEVO_PREFILTER_INVALID_CANDIDATES="$PREFILTER_INVALID_CANDIDATES" \
    PIEVO_MAX_PREFILTER_REJECTIONS="$MAX_PREFILTER_REJECTIONS" \
    PIEVO_MAX_PREFILTER_EMPTY_POOLS="$MAX_PREFILTER_EMPTY_POOLS" \
    PIEVO_RADP_SEARCH_STRATEGY="$RADP_SEARCH_STRATEGY" \
    PIEVO_COMMON_GENERATOR="$method_common_generator" \
    PIEVO_AFE_SCHEMA_VERSION=v8 \
    PIEVO_RADP_WIDE_SEED_BANK="$RADP_WIDE_SEED_BANK" \
    PIEVO_DISABLE_PLOTS=1 \
    PIEVO_CORRUPTION_MODE="$CORRUPTION_MODE" \
    PIEVO_CORRUPTION_CALLS="$CORRUPTION_CALLS" \
    PIEVO_CORRUPTION_OFFSET="$CORRUPTION_OFFSET" \
    PIEVO_SOURCE_GATE_SCORE_GE="$SOURCE_GATE_SCORE_GE" \
  )
  set +e
  if [[ "$DRY_RUN" == "1" ]]; then
    env -u ZJU_TASK_API_KEY -u OPENAI_API_KEY \
      "${runner_env[@]}" "$ROOT/experiments/run_matched_qwen235b_task.sh"
  else
    env "${runner_env[@]}" "$ROOT/experiments/run_matched_qwen235b_task.sh"
  fi
  method_status=$?
  set -e
  if [[ "$method_status" -ne 0 ]]; then
    METHOD_FAILURE_STATUS="$method_status"
    METHOD_FAILURE_NAME="$method"
    break
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  if [[ "$METHOD_FAILURE_STATUS" -ne 0 ]]; then
    echo "Paired MBO development dry-run failed for method=$METHOD_FAILURE_NAME" >&2
    exit "$METHOD_FAILURE_STATUS"
  fi
  echo "Paired MBO development dry-run passed; no API key was read and no model request was made."
else
  set +e
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
    "$ROOT/experiments/audit_official_mbo_development_screen.py" \
    --root "$OUT_ROOT" \
    --seed "$SEED" \
    --budget "$BUDGET" \
    --mode "$SCREEN_MODE" \
    --token-ceiling "$TOKEN_CEILING" \
    --output "$OUT_ROOT/official_mbo_development_screen_audit.json"
  audit_status=$?
  set -e
  if [[ "$METHOD_FAILURE_STATUS" -ne 0 ]]; then
    echo "Paired screen stopped after method=$METHOD_FAILURE_NAME failed; failure audit was written." >&2
    exit "$METHOD_FAILURE_STATUS"
  fi
  if [[ "$audit_status" -ne 0 ]]; then
    echo "Paired screen completed but failed its audit." >&2
    exit "$audit_status"
  fi
  echo "Paired MBO development screen completed: seed=$SEED, primary_per_method=$BUDGET, output=$OUT_ROOT"
  echo "Interpret as a controlled official-MBO development diagnostic only; it does not unlock formal or external evaluation."
fi
