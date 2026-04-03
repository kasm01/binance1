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

  local line k v

  while IFS= read -r line || [[ -n "$line" ]]; do
    # boş / yorum
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    # export prefix
    [[ "$line" == export\ * ]] && line="${line#export }"

    # "=" yoksa geç
    [[ "$line" == *=* ]] || continue

    k="${line%%=*}"
    v="${line#*=}"

    # trim key
    k="${k#"${k%%[![:space:]]*}"}"
    k="${k%"${k##*[![:space:]]}"}"

    [[ -z "$k" ]] && continue
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    # mevcut env varsa override etme
    if [[ -n "${!k+x}" ]]; then
      continue
    fi

    # trim value
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"

    # tek/double quote soy
    if [[ ${#v} -ge 2 ]]; then
      if [[ "${v:0:1}" == '"' && "${v: -1}" == '"' ]]; then
        v="${v:1:${#v}-2}"
      elif [[ "${v:0:1}" == "'" && "${v: -1}" == "'" ]]; then
        v="${v:1:${#v}-2}"
      fi
    fi

    export "$k=$v"
  done < "$env_file"
}
load_env_fill_only ".env"

# -----------------------------
# Dirs + lock
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
# Streams
# -----------------------------
SIGNALS_STREAM="${SIGNALS_STREAM:-signals_stream}"
CANDIDATES_STREAM="${CANDIDATES_STREAM:-candidates_stream}"
TOP5_STREAM="${TOP5_STREAM:-top5_stream}"
TRADE_INTENTS_STREAM="${TRADE_INTENTS_STREAM:-trade_intents_stream}"
EXEC_EVENTS_STREAM="${EXEC_EVENTS_STREAM:-exec_events_stream}"

# -----------------------------
# Consumer groups
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
# Redis bootstrap
# -----------------------------
redis_xgroup_create() {
  local stream="$1" group="$2" start_id="${3:-$}"
  need redis-cli || return 0
  redis-cli -n "$REDIS_DB" XGROUP CREATE "$stream" "$group" "$start_id" MKSTREAM >/dev/null 2>&1 || true
}

bootstrap_redis() {
  redis_xgroup_create "$CANDIDATES_STREAM" "$TOPSEL_GROUP" "$TOPSEL_GROUP_START_ID"
  redis_xgroup_create "$TOP5_STREAM" "$MASTER_GROUP" "$MASTER_GROUP_START_ID"
  redis_xgroup_create "$TRADE_INTENTS_STREAM" "$BRIDGE_GROUP" "$BRIDGE_GROUP_START_ID"

  need redis-cli || return 0
  redis-cli -n "$REDIS_DB" XADD "$SIGNALS_STREAM" "*" ping "1" >/dev/null 2>&1 || true
  redis-cli -n "$REDIS_DB" XADD "$EXEC_EVENTS_STREAM" "*" ping "1" >/dev/null 2>&1 || true
}
bootstrap_redis

# -----------------------------
# Helpers: pidfile heal / cleanup
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
# Process health helpers
# -----------------------------
proc_pattern() {
  local name="$1"
  case "$name" in
    scanner_w1) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w1|w1')" ;;
    scanner_w2) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w2|w2')" ;;
    scanner_w3) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w3|w3')" ;;
    scanner_w4) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w4|w4')" ;;
    scanner_w5) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w5|w5')" ;;
    scanner_w6) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w6|w6')" ;;
    scanner_w7) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w7|w7')" ;;
    scanner_w8) echo "orchestration/scanners/worker_stub.py.*WORKER_ID.?=.?(w8|w8')" ;;
    aggregator) echo "orchestration/aggregator/run_aggregator.py" ;;
    top_selector) echo "orchestration/selector/top_selector.py" ;;
    master_executor) echo "orchestration/executor/master_executor.py" ;;
    intent_bridge) echo "orchestration/executor/intent_bridge.py" ;;
    *) echo "$name" ;;
  esac
}

is_proc_alive() {
  local pat="$1"
  pgrep -f "$pat" >/dev/null 2>&1
}

wait_proc_alive() {
  local pat="$1"
  local seconds_total="${2:-8}"
  local sleep_step="${3:-1}"
  local elapsed=0

  while (( elapsed < seconds_total )); do
    if is_proc_alive "$pat"; then
      return 0
    fi
    sleep "$sleep_step"
    elapsed=$((elapsed + sleep_step))
  done
  return 1
}

assert_proc_running() {
  local name="$1"
  local pat
  pat="$(proc_pattern "$name")"
  if wait_proc_alive "$pat" 8 1; then
    local pid
    pid="$(pgrep -f "$pat" | head -n1 || true)"
    [[ -n "${pid:-}" ]] && echo "$pid" > "run/${name}.pid"
    echo "[OK] ${name} running pid=${pid:-na}"
    return 0
  fi

  echo "[FAIL] ${name} is not running (pattern=${pat})"
  echo "[FAIL] Check log: logs/orch/${name}.log"
  return 1
}
# -----------------------------
# Backpressure Guard
# -----------------------------
BP_ENABLED="${BP_ENABLED:-1}"
BP_THROTTLE_KEY="${BP_THROTTLE_KEY:-scanner:throttle_factor}"

BP_EXEC_XLEN_HIGH="${BP_EXEC_XLEN_HIGH:-5000}"
BP_EXEC_XLEN_LOW="${BP_EXEC_XLEN_LOW:-2000}"
BP_FACTOR_MIN="${BP_FACTOR_MIN:-1}"
BP_FACTOR_MAX="${BP_FACTOR_MAX:-16}"
BP_FACTOR_DEFAULT="${BP_FACTOR_DEFAULT:-1}"
BP_THROTTLE_TTL_SEC="${BP_THROTTLE_TTL_SEC:-120}"
BP_FACTOR_STEP_UP="${BP_FACTOR_STEP_UP:-2}"
BP_FACTOR_STEP_DOWN="${BP_FACTOR_STEP_DOWN:-2}"

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

  cur="$(echo "$cur" | awk '{print int($1)}' 2>/dev/null || echo "$BP_FACTOR_DEFAULT")"
  cur="$(clamp_int "$cur" "$BP_FACTOR_MIN" "$BP_FACTOR_MAX")"

  local next="$cur"

  if [[ "$xlen" -ge "$BP_EXEC_XLEN_HIGH" ]]; then
    next=$(( cur * BP_FACTOR_STEP_UP ))
    next="$(clamp_int "$next" "$BP_FACTOR_MIN" "$BP_FACTOR_MAX")"
    echo "[BP] exec_events_stream XLEN=$xlen >= $BP_EXEC_XLEN_HIGH -> throttle $cur -> $next (key=$BP_THROTTLE_KEY)"
    redis_set_throttle "$next"
    return 0
  fi

  if [[ "$xlen" -le "$BP_EXEC_XLEN_LOW" ]]; then
    if [[ "$cur" -gt "$BP_FACTOR_DEFAULT" ]]; then
      next=$(( cur / BP_FACTOR_STEP_DOWN ))
      [[ "$next" -lt "$BP_FACTOR_DEFAULT" ]] && next="$BP_FACTOR_DEFAULT"
      next="$(clamp_int "$next" "$BP_FACTOR_MIN" "$BP_FACTOR_MAX")"
      echo "[BP] exec_events_stream XLEN=$xlen <= $BP_EXEC_XLEN_LOW -> throttle $cur -> $next (key=$BP_THROTTLE_KEY)"
      redis_set_throttle "$next"
    else
      redis_set_throttle "$BP_FACTOR_DEFAULT"
    fi
    return 0
  fi

  echo "[BP] exec_events_stream XLEN=$xlen in mid-zone -> keep throttle=$cur (refresh TTL)"
  redis_set_throttle "$cur"
}
apply_backpressure_guard

# -----------------------------
# Stuck consumer auto reset
# -----------------------------
SUP_AUTORESET_ENABLED="${SUP_AUTORESET_ENABLED:-1}"
SUP_AUTORESET_IDLE_MS="${SUP_AUTORESET_IDLE_MS:-120000}"
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
  autoreset_stream_group "$TRADE_INTENTS_STREAM" "$BRIDGE_GROUP"
  autoreset_stream_group "$TOP5_STREAM" "$MASTER_GROUP"
}
autoreset_best_effort

# -----------------------------
# DRY_RUN reset policy
# -----------------------------
DRYRUN_RESET_STATE="${DRYRUN_RESET_STATE:-1}"

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
# Optional sterile mode
# -----------------------------
STERILE="${STERILE:-0}"
if is_truthy "$STERILE"; then
  echo "[MODE] STERILE=1 -> starting only master_executor + intent_bridge"

  heal_pidfile "master_executor" "$(proc_pattern master_executor)"
  heal_pidfile "intent_bridge" "$(proc_pattern intent_bridge)"

  start_one "master_executor" "$PY" -u ./orchestration/executor/master_executor.py
  start_one "intent_bridge" "$PY" -u ./orchestration/executor/intent_bridge.py

  assert_proc_running "master_executor"
  assert_proc_running "intent_bridge"

  echo
  echo "Sterile start commands issued."
  echo "Tail logs: tail -n 200 -f logs/orch/*.log"
  exit 0
fi

# -----------------------------
# SCANNERS
# -----------------------------
SYMBOLS_24="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,AVAXUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,OPUSDT,ARBUSDT,NEARUSDT,HBARUSDT,INJUSDT}"
WORKER_INTERVAL="${WORKER_INTERVAL:-5m}"
SCANNER_MODE="${SCANNER_MODE:-fast}"

WORKER_MAX_SPREAD_PCT="${WORKER_MAX_SPREAD_PCT:-0.0010}"
WORKER_MAX_ATR_PCT="${WORKER_MAX_ATR_PCT:-0.0300}"
WORKER_MIN_CONF="${WORKER_MIN_CONF:-0.45}"

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

for n in 1 2 3 4 5 6 7 8; do
  wid="w${n}"
  name="scanner_${wid}"
  syms="$(pick_worker_symbols "$n")"

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
    "$PY" -u ./orchestration/scanners/worker_stub.py
done
# -----------------------------
# Downstream singletons
# -----------------------------
heal_pidfile "aggregator" "$(proc_pattern aggregator)"
heal_pidfile "top_selector" "$(proc_pattern top_selector)"
heal_pidfile "master_executor" "$(proc_pattern master_executor)"
heal_pidfile "intent_bridge" "$(proc_pattern intent_bridge)"

start_one "aggregator" "$PY" -u ./orchestration/aggregator/run_aggregator.py
start_one "top_selector" "$PY" -u ./orchestration/selector/top_selector.py
start_one "master_executor" "$PY" -u ./orchestration/executor/master_executor.py
start_one "intent_bridge" "$PY" -u ./orchestration/executor/intent_bridge.py

# hard verify singleton processes
assert_proc_running "aggregator"
assert_proc_running "top_selector"
assert_proc_running "master_executor"
assert_proc_running "intent_bridge"

# -----------------------------
# Readiness
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
    local raw
    raw="$(redis-cli -n "$REDIS_DB" --raw XINFO GROUPS "$stream" 2>/dev/null || true)"
    if [[ -n "${raw:-}" ]]; then
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

  assert_proc_running "aggregator"
  assert_proc_running "top_selector"
  assert_proc_running "master_executor"
  assert_proc_running "intent_bridge"

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

  if is_truthy "$BP_ENABLED"; then
    cur="$(redis_get_int "$BP_THROTTLE_KEY")"; cur="${cur:-$BP_FACTOR_DEFAULT}"
    xlen="$(redis_xlen "$EXEC_EVENTS_STREAM")"
    echo "[BP] status: key=$BP_THROTTLE_KEY factor=$cur XLEN($EXEC_EVENTS_STREAM)=$xlen (hi=$BP_EXEC_XLEN_HIGH lo=$BP_EXEC_XLEN_LOW)"
  fi
fi

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"
