#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/orch_lib.sh

# -----------------------------
# Load .env into THIS shell (export to children) WITHOUT overriding existing env
# CLI/env vars should win over .env
# -----------------------------
load_env_fill_only() {
  local env_file="${1:-.env}"
  [[ -f "$env_file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    line="${line#export }"

    local k="${line%%=*}"
    local v="${line#*=}"

    k="$(echo "$k" | xargs)"
    [[ -z "$k" ]] && continue
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    # do not override already-set env
    if [[ -n "${!k+x}" ]]; then
      continue
    fi

    v="$(echo "$v" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [[ "$v" =~ ^\".*\"$ ]]; then
      v="${v:1:${#v}-2}"
    elif [[ "$v" =~ ^\'.*\'$ ]]; then
      v="${v:1:${#v}-2}"
    fi

    export "$k=$v"
  done < "$env_file"
}

load_env_fill_only ".env"

# -----------------------------
# Dirs + lock (avoid double-start races)
# -----------------------------
mkdir -p run logs/orch
LOCK_FILE="run/orch_start.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[START][WARN] another orch_start is running (lock=$LOCK_FILE) -> exit"
  exit 0
fi

need() { command -v "$1" >/dev/null 2>&1; }

REDIS_DB="${REDIS_DB:-0}"
PY="${PY:-python}"

# -----------------------------
# Streams (prefer .env)
# -----------------------------
SIGNALS_STREAM="${SIGNALS_STREAM:-signals_stream}"
CANDIDATES_STREAM="${CANDIDATES_STREAM:-candidates_stream}"
TOP5_STREAM="${TOP5_STREAM:-top5_stream}"
TRADE_INTENTS_STREAM="${TRADE_INTENTS_STREAM:-trade_intents_stream}"
EXEC_EVENTS_STREAM="${EXEC_EVENTS_STREAM:-exec_events_stream}"

# -----------------------------
# Hardening: pin consumer groups (deterministic)
# -----------------------------
export TOPSEL_GROUP="${TOPSEL_GROUP:-topsel_g}"
export TOPSEL_GROUP_START_ID="${TOPSEL_GROUP_START_ID:-$}"
export TOPSEL_CONSUMER="${TOPSEL_CONSUMER:-topsel_1}"

export BRIDGE_GROUP="${BRIDGE_GROUP:-bridge_g}"
export BRIDGE_GROUP_START_ID="${BRIDGE_GROUP_START_ID:-$}"
export BRIDGE_CONSUMER="${BRIDGE_CONSUMER:-bridge_1}"

export MASTER_GROUP="${MASTER_GROUP:-master_exec_g}"
export MASTER_GROUP_START_ID="${MASTER_GROUP_START_ID:-$}"
export MASTER_CONSUMER="${MASTER_CONSUMER:-master_1}"

# -----------------------------
# Redis bootstrap: create streams+groups (idempotent)
# -----------------------------
redis_xgroup_create() {
  local stream="$1" group="$2" start_id="${3:-$}"
  need redis-cli || return 0
  redis-cli -n "$REDIS_DB" XGROUP CREATE "$stream" "$group" "$start_id" MKSTREAM >/dev/null 2>&1 || true
}

bootstrap_redis() {
  # pipeline streams
  redis_xgroup_create "$CANDIDATES_STREAM" "$TOPSEL_GROUP" "$TOPSEL_GROUP_START_ID"
  redis_xgroup_create "$TOP5_STREAM"       "$MASTER_GROUP" "$MASTER_GROUP_START_ID"
  redis_xgroup_create "$TRADE_INTENTS_STREAM" "$BRIDGE_GROUP" "$BRIDGE_GROUP_START_ID"

  # ensure these streams exist (harmless)
  need redis-cli || return 0
  redis-cli -n "$REDIS_DB" XADD "$SIGNALS_STREAM" "*" ping "1" >/dev/null 2>&1 || true
  redis-cli -n "$REDIS_DB" XADD "$EXEC_EVENTS_STREAM" "*" ping "1" >/dev/null 2>&1 || true
}
bootstrap_redis

# -----------------------------
# Helpers: pidfile heal / cleanup (prevents stale pid alarms)
# -----------------------------
heal_pidfile() {
  local name="$1" pat="$2"
  local pidfile="run/${name}.pid"
  [[ -f "$pidfile" ]] || return 0

  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "${pid:-}" ]] || ! kill -0 "$pid" 2>/dev/null; then
    local newpid
    newpid="$(pgrep -f "$pat" | head -n1 || true)"
    if [[ -n "${newpid:-}" ]]; then
      echo "$newpid" > "$pidfile"
      echo "[START][WARN] ${name} pidfile stale -> healed (old=${pid:-na} new=${newpid})"
    else
      rm -f "$pidfile" || true
      echo "[START][WARN] ${name} pidfile stale -> cleaned (old=${pid:-na})"
    fi
  fi
}

# -----------------------------
# Backpressure Guard (exec_events_stream)
#   - sets scanner:throttle_factor (workers read this and slow down)
#   - best-effort: only on start, and can be re-run via orch service restart
# -----------------------------
BP_ENABLED="${BP_ENABLED:-1}"
BP_THROTTLE_KEY="${BP_THROTTLE_KEY:-scanner:throttle_factor}"

BP_EXEC_XLEN_HIGH="${BP_EXEC_XLEN_HIGH:-5000}"   # trigger slow down
BP_EXEC_XLEN_LOW="${BP_EXEC_XLEN_LOW:-2000}"     # allow speed up (hysteresis)
BP_FACTOR_MIN="${BP_FACTOR_MIN:-1}"
BP_FACTOR_MAX="${BP_FACTOR_MAX:-16}"
BP_FACTOR_DEFAULT="${BP_FACTOR_DEFAULT:-1}"
BP_THROTTLE_TTL_SEC="${BP_THROTTLE_TTL_SEC:-120}"  # keep alive window
BP_FACTOR_STEP_UP="${BP_FACTOR_STEP_UP:-2}"         # multiply on spike (2x)
BP_FACTOR_STEP_DOWN="${BP_FACTOR_STEP_DOWN:-2}"     # divide on relief (2x)

redis_get_int() {
  local key="$1"
  need redis-cli || { echo ""; return 0; }
  local v
  v="$(redis-cli -n "$REDIS_DB" GET "$key" 2>/dev/null || true)"
  [[ -n "${v:-}" ]] || { echo ""; return 0; }
  echo "$v" | tr -d '\r' | awk '{print $1}'
}

redis_set_throttle() {
  local factor="$1"
  need redis-cli || return 0
  # set with TTL so it auto-expires back to normal if nothing refreshes it
  redis-cli -n "$REDIS_DB" SET "$BP_THROTTLE_KEY" "$factor" EX "$BP_THROTTLE_TTL_SEC" >/dev/null 2>&1 || true
}

redis_xlen() {
  local stream="$1"
  need redis-cli || { echo "0"; return 0; }
  redis-cli -n "$REDIS_DB" XLEN "$stream" 2>/dev/null | tr -d '\r' | awk '{print $1}' || echo "0"
}

clamp_int() {
  local x="$1" lo="$2" hi="$3"
  if [[ "$x" -lt "$lo" ]]; then echo "$lo"; return 0; fi
  if [[ "$x" -gt "$hi" ]]; then echo "$hi"; return 0; fi
  echo "$x"
}

apply_backpressure_guard() {
  is_truthy "$BP_ENABLED" || return 0
  need redis-cli || return 0

  local xlen
  xlen="$(redis_xlen "$EXEC_EVENTS_STREAM")"
  [[ -n "${xlen:-}" ]] || xlen="0"

  local cur
  cur="$(redis_get_int "$BP_THROTTLE_KEY")"
  [[ -n "${cur:-}" ]] || cur="$BP_FACTOR_DEFAULT"

  # normalize cur
  cur="$(echo "$cur" | awk '{print int($1)}' 2>/dev/null || echo "$BP_FACTOR_DEFAULT")"
  cur="$(clamp_int "$cur" "$BP_FACTOR_MIN" "$BP_FACTOR_MAX")"

  local next="$cur"

  if [[ "$xlen" -ge "$BP_EXEC_XLEN_HIGH" ]]; then
    # step up (multiply)
    next=$(( cur * BP_FACTOR_STEP_UP ))
    next="$(clamp_int "$next" "$BP_FACTOR_MIN" "$BP_FACTOR_MAX")"
    echo "[BP] exec_events_stream XLEN=$xlen >= $BP_EXEC_XLEN_HIGH -> throttle $cur -> $next (key=$BP_THROTTLE_KEY)"
    redis_set_throttle "$next"
    return 0
  fi

  if [[ "$xlen" -le "$BP_EXEC_XLEN_LOW" ]]; then
    # step down (divide)
    if [[ "$cur" -gt "$BP_FACTOR_DEFAULT" ]]; then
      next=$(( cur / BP_FACTOR_STEP_DOWN ))
      [[ "$next" -lt "$BP_FACTOR_DEFAULT" ]] && next="$BP_FACTOR_DEFAULT"
      next="$(clamp_int "$next" "$BP_FACTOR_MIN" "$BP_FACTOR_MAX")"
      echo "[BP] exec_events_stream XLEN=$xlen <= $BP_EXEC_XLEN_LOW -> throttle $cur -> $next (key=$BP_THROTTLE_KEY)"
      redis_set_throttle "$next"
    else
      # keep refreshed at default so workers converge fast
      redis_set_throttle "$BP_FACTOR_DEFAULT"
    fi
    return 0
  fi

  # mid-zone: keep current but refresh TTL
  echo "[BP] exec_events_stream XLEN=$xlen in mid-zone -> keep throttle=$cur (refresh TTL)"
  redis_set_throttle "$cur"
}

apply_backpressure_guard

# -----------------------------
# Stuck consumer auto reset (best effort)
#   - claims idle pending messages so they don't get stuck forever
#   - safe to run even if no pending
# -----------------------------
SUP_AUTORESET_ENABLED="${SUP_AUTORESET_ENABLED:-1}"
SUP_AUTORESET_IDLE_MS="${SUP_AUTORESET_IDLE_MS:-120000}"  # 120s
SUP_AUTORESET_COUNT="${SUP_AUTORESET_COUNT:-50}"
SUP_AUTORESET_CONSUMER="${SUP_AUTORESET_CONSUMER:-orch_autoreset}"

autoreset_stream_group() {
  local stream="$1" group="$2"
  need redis-cli || return 0
  redis-cli -n "$REDIS_DB" \
    XAUTOCLAIM "$stream" "$group" "$SUP_AUTORESET_CONSUMER" "$SUP_AUTORESET_IDLE_MS" 0 \
    COUNT "$SUP_AUTORESET_COUNT" >/dev/null 2>&1 || true
}

autoreset_best_effort() {
  is_truthy "$SUP_AUTORESET_ENABLED" || return 0
  # These are your critical groups
  autoreset_stream_group "$TRADE_INTENTS_STREAM" "$BRIDGE_GROUP"
  autoreset_stream_group "$TOP5_STREAM" "$MASTER_GROUP"
}
autoreset_best_effort

# -----------------------------
# DRY_RUN reset policy (configurable)
#   - DRYRUN_RESET_STATE=1  => allow reset (default)
#   - DRYRUN_RESET_STATE=0  => never reset
#   - If BRIDGE_DRYRUN_WRITE_STATE=1, default behavior is to NOT reset (keep state)
# -----------------------------
DRYRUN_RESET_STATE="${DRYRUN_RESET_STATE:-1}"

# If user explicitly wants dry-run write-state, default to preserving state unless overridden.
if is_truthy "${BRIDGE_DRYRUN_WRITE_STATE:-0}"; then
  DRYRUN_RESET_STATE="${DRYRUN_RESET_STATE_OVERRIDE_WHEN_WRITE_STATE:-0}"
fi

if is_truthy "${DRY_RUN:-1}"; then
  if is_truthy "$DRYRUN_RESET_STATE"; then
    if ! any_orch_running; then
      echo "[START] DRY_RUN -> resetting open_positions_state + positions:* (dry-run safety reset)"
      need redis-cli || true

      redis-cli -n "$REDIS_DB" DEL "${BRIDGE_STATE_KEY:-open_positions_state}" >/dev/null 2>&1 || true
      POS_PREFIX="${POSITION_KEY_PREFIX:-positions}"
      redis-cli -n "$REDIS_DB" --scan --pattern "${POS_PREFIX}:*" \
        | xargs -r -n 200 redis-cli -n "$REDIS_DB" DEL >/dev/null 2>&1 || true
    else
      echo "[SKIP] DRY_RUN reset (processes already running)"
    fi
  else
    echo "[SKIP] DRY_RUN reset disabled (DRYRUN_RESET_STATE=0)"
  fi
fi
# -----------------------------
# Optional: sterile mode (only master + bridge)
# -----------------------------
STERILE="${STERILE:-0}"
if is_truthy "$STERILE"; then
  echo "[MODE] STERILE=1 -> starting only master_executor + intent_bridge"
  heal_pidfile "master_executor" "orchestration/executor/master_executor.py"
  heal_pidfile "intent_bridge"   "orchestration/executor/intent_bridge.py"

  start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py
  start_one "intent_bridge"  "$PY" -u orchestration/executor/intent_bridge.py

  echo
  echo "Sterile start commands issued."
  echo "Tail logs: tail -n 200 -f logs/orch/*.log"
  exit 0
fi

# -----------------------------
# SCANNERS: symbols -> 8 workers (3 symbols each)
# Priority:
#   1) Wn_SYMBOLS override
#   2) SYMBOLS_24 split (can be >24; we'll just take sequential triplets)
# -----------------------------
SYMBOLS_24="${SYMBOLS_24:-BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,MATICUSDT,DOTUSDT,LTCUSDT,TRXUSDT,ATOMUSDT,OPUSDT,ARBUSDT,APTUSDT,INJUSDT,SUIUSDT,FILUSDT,NEARUSDT,ETCUSDT,UNIUSDT,AAVEUSDT}"
WORKER_INTERVAL="${WORKER_INTERVAL:-5m}"
SCANNER_MODE="${SCANNER_MODE:-fast}"

WORKER_MAX_SPREAD_PCT="${WORKER_MAX_SPREAD_PCT:-0.0010}"
WORKER_MAX_ATR_PCT="${WORKER_MAX_ATR_PCT:-0.0300}"
WORKER_MIN_CONF="${WORKER_MIN_CONF:-0.45}"

# worker throttle envs (read by updated worker_stub.py)
WORKER_THROTTLE_KEY="${WORKER_THROTTLE_KEY:-$BP_THROTTLE_KEY}"
WORKER_THROTTLE_MIN="${WORKER_THROTTLE_MIN:-1}"
WORKER_THROTTLE_MAX="${WORKER_THROTTLE_MAX:-20}"
WORKER_THROTTLE_POLL_EVERY="${WORKER_THROTTLE_POLL_EVERY:-10}"
WORKER_THROTTLE_JITTER="${WORKER_THROTTLE_JITTER:-0.02}"

IFS=',' read -r -a symarr <<<"$SYMBOLS_24"

_join_nonempty_csv() {
  local out=""
  local x
  for x in "$@"; do
    x="$(echo "$x" | xargs)"
    [[ -z "$x" ]] && continue
    if [[ -z "$out" ]]; then out="$x"; else out="${out},${x}"; fi
  done
  echo "$out"
}

pick_worker_symbols() {
  local n="$1"
  local var="W${n}_SYMBOLS"
  local override="${!var:-}"
  if [[ -n "${override:-}" ]]; then
    echo "${override}"
    return 0
  fi

  local idx=$(( (n-1) * 3 ))
  _join_nonempty_csv "${symarr[$idx]:-}" "${symarr[$((idx+1))]:-}" "${symarr[$((idx+2))]:-}"
}

# Start scanners
for n in 1 2 3 4 5 6 7 8; do
  wid="w${n}"
  name="scanner_${wid}"
  syms="$(pick_worker_symbols "$n")"

  # if worker gets empty (not enough symbols), skip
  if [[ -z "${syms:-}" ]]; then
    echo "[SKIP] ${name} (no symbols assigned)"
    continue
  fi

  start_one "$name" env \
    WORKER_ID="$wid" \
    WORKER_SYMBOLS="$syms" \
    WORKER_INTERVAL="$WORKER_INTERVAL" \
    SCANNER_MODE="$SCANNER_MODE" \
    WORKER_PUBLISH_HOLD=0 \
    WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" \
    WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" \
    WORKER_MIN_CONF="$WORKER_MIN_CONF" \
    WORKER_THROTTLE_KEY="$WORKER_THROTTLE_KEY" \
    WORKER_THROTTLE_MIN="$WORKER_THROTTLE_MIN" \
    WORKER_THROTTLE_MAX="$WORKER_THROTTLE_MAX" \
    WORKER_THROTTLE_POLL_EVERY="$WORKER_THROTTLE_POLL_EVERY" \
    WORKER_THROTTLE_JITTER="$WORKER_THROTTLE_JITTER" \
    "$PY" -u orchestration/scanners/worker_stub.py
done

# -----------------------------
# Downstream singletons (ordered)
# -----------------------------
heal_pidfile "aggregator"      "orchestration/aggregator/run_aggregator.py"
heal_pidfile "top_selector"    "orchestration/selector/top_selector.py"
heal_pidfile "master_executor" "orchestration/executor/master_executor.py"
heal_pidfile "intent_bridge"   "orchestration/executor/intent_bridge.py"

start_one "aggregator"      "$PY" -u orchestration/aggregator/run_aggregator.py
start_one "top_selector"    "$PY" -u orchestration/selector/top_selector.py
start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py
start_one "intent_bridge"   "$PY" -u orchestration/executor/intent_bridge.py

# -----------------------------
# Readiness (best effort)
# -----------------------------
_last_stream_id() {
  local stream="$1"
  need redis-cli || { echo ""; return 0; }
  redis-cli -n "$REDIS_DB" --raw XINFO STREAM "$stream" 2>/dev/null \
    | awk 'BEGIN{FS="\n"}{for(i=1;i<=NF;i++){if($i=="last-generated-id"){print $(i+1); exit}}}' \
    | tr -d '\r' || true
}

_wait_stream_advance_by_id() {
  local stream="$1"
  local seconds_total="${2:-6}"
  local sleep_step="${3:-1}"

  need redis-cli || return 0
  local a b
  a="$(_last_stream_id "$stream")"; a="${a:-}"
  local elapsed=0

  while (( elapsed < seconds_total )); do
    sleep "$sleep_step"
    elapsed=$((elapsed + sleep_step))
    b="$(_last_stream_id "$stream")"
    if [[ -n "${b:-}" ]] && [[ "${b:-}" != "${a:-}" ]]; then
      return 0
    fi
  done
  return 1
}

_wait_group_health() {
  local stream="$1" group="$2"
  local seconds_total="${3:-6}"
  local sleep_step="${4:-1}"

  need redis-cli || return 0
  local elapsed=0

  while (( elapsed < seconds_total )); do
    # Use --raw for stable parsing: fields are alternating key/value lines
    local raw
    raw="$(redis-cli -n "$REDIS_DB" --raw XINFO GROUPS "$stream" 2>/dev/null || true)"
    if [[ -n "${raw:-}" ]]; then
      # Extract pending+lag for the given group
      local pending lag
      pending="$(printf "%s\n" "$raw" | awk -v g="$group" '
        $0=="name"{getline; name=$0}
        $0=="pending"{getline; p=$0}
        $0=="lag"{getline; l=$0; if(name==g){print p; exit}}
      ' | tr -d '\r' | head -n1 || true)"

      lag="$(printf "%s\n" "$raw" | awk -v g="$group" '
        $0=="name"{getline; name=$0}
        $0=="lag"{getline; l=$0; if(name==g){print l; exit}}
      ' | tr -d '\r' | head -n1 || true)"

      if [[ -n "${pending:-}" ]] && [[ -n "${lag:-}" ]]; then
        if [[ "${pending}" == "0" && "${lag}" == "0" ]]; then
          return 0
        fi
      fi
    fi

    sleep "$sleep_step"
    elapsed=$((elapsed + sleep_step))
  done

  return 1
}

WAIT_READY="${WAIT_READY:-1}"
if is_truthy "$WAIT_READY"; then
  echo
  echo "=== Readiness check (best effort) ==="

  if _wait_stream_advance_by_id "$SIGNALS_STREAM" 6 1; then
    echo "[OK] ${SIGNALS_STREAM} advanced (id)"
  else
    echo "[WARN] ${SIGNALS_STREAM} did not advance in time (might still be OK)"
  fi

  if _wait_stream_advance_by_id "$TRADE_INTENTS_STREAM" 6 1; then
    echo "[OK] ${TRADE_INTENTS_STREAM} advanced (id)"
  else
    echo "[WARN] ${TRADE_INTENTS_STREAM} did not advance in time (might still be OK)"
  fi

  if _wait_stream_advance_by_id "$EXEC_EVENTS_STREAM" 6 1; then
    echo "[OK] ${EXEC_EVENTS_STREAM} advanced (id)"
  else
    echo "[WARN] ${EXEC_EVENTS_STREAM} did not advance in time (might still be OK)"
  fi

  if _wait_group_health "$TRADE_INTENTS_STREAM" "$BRIDGE_GROUP" 6 1; then
    echo "[OK] ${TRADE_INTENTS_STREAM} group healthy (group=$BRIDGE_GROUP pending=0 lag=0)"
  else
    echo "[WARN] ${TRADE_INTENTS_STREAM} group not healthy/visible yet (group=$BRIDGE_GROUP)"
  fi

  if _wait_group_health "$TOP5_STREAM" "$MASTER_GROUP" 6 1; then
    echo "[OK] ${TOP5_STREAM} group healthy (group=$MASTER_GROUP pending=0 lag=0)"
  else
    echo "[WARN] ${TOP5_STREAM} group not healthy/visible yet (group=$MASTER_GROUP)"
  fi

  # print backpressure state summary
  if is_truthy "$BP_ENABLED"; then
    cur="$(redis_get_int "$BP_THROTTLE_KEY")"; cur="${cur:-$BP_FACTOR_DEFAULT}"
    xlen="$(redis_xlen "$EXEC_EVENTS_STREAM")"
    echo "[BP] status: key=$BP_THROTTLE_KEY factor=$cur XLEN($EXEC_EVENTS_STREAM)=$xlen (hi=$BP_EXEC_XLEN_HIGH lo=$BP_EXEC_XLEN_LOW)"
  fi
fi

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"
