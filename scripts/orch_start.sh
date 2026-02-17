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
    # skip empty/comment
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    # accept KEY=VALUE (optionally starting with "export ")
    line="${line#export }"

    # split on first '='
    local k="${line%%=*}"
    local v="${line#*=}"

    # trim key
    k="$(echo "$k" | xargs)"
    [[ -z "$k" ]] && continue

    # if key already set in environment, do not override
    if [[ -n "${!k+x}" ]]; then
      continue
    fi

    # trim surrounding whitespace of value
    v="$(echo "$v" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # strip surrounding quotes (single or double)
    if [[ "$v" =~ ^\".*\"$ ]]; then
      v="${v:1:${#v}-2}"
    elif [[ "$v" =~ ^\'.*\'$ ]]; then
      v="${v:1:${#v}-2}"
    fi

    export "$k=$v"
  done < "$env_file"
}

load_env_fill_only ".env"

# Optional legacy loader: only if it also respects "do not override"
# (If your load_env.sh overrides, KEEP THIS DISABLED)
# if [[ -x "./scripts/load_env.sh" ]] && [[ -f ".env" ]]; then
#   ./scripts/load_env.sh .env >/dev/null 2>&1 || true
# fi

# -----------------------------
# Ensure dirs exist
# -----------------------------
mkdir -p run logs/orch

# Redis DB (default 0)
REDIS_DB="${REDIS_DB:-0}"

# -----------------------------
# Hardening: pin Redis stream consumer groups (deterministic)
# -----------------------------
# IntentBridge group defaults to "bridge_g" in code, but we pin it here to avoid surprises.
export BRIDGE_GROUP="${BRIDGE_GROUP:-bridge_g}"
export BRIDGE_GROUP_START_ID="${BRIDGE_GROUP_START_ID:-$}"
export BRIDGE_CONSUMER="${BRIDGE_CONSUMER:-bridge_1}"

# MasterExecutor group pin (optional; safe defaults)
export MASTER_GROUP="${MASTER_GROUP:-master_exec_g}"
export MASTER_GROUP_START_ID="${MASTER_GROUP_START_ID:-$}"
export MASTER_CONSUMER="${MASTER_CONSUMER:-master_1}"

# -----------------------------
# Helpers: pidfile heal / cleanup
# -----------------------------
heal_pidfile() {
    # Usage: heal_pidfile <name> <pattern>
    local name="$1"
    local pat="$2"
    local pidfile="run/${name}.pid"

    if [[ -f "$pidfile" ]]; then
        local pid
        pid="$(cat "$pidfile" 2>/dev/null || true)"

        # empty or dead pid -> attempt heal by pgrep, else cleanup
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
    fi
}

# -----------------------------
# Helpers: stream readiness
# -----------------------------
_last_stream_id() {
    # Usage: _last_stream_id <stream>
    # Returns last entry id or empty.
    local stream="$1"
    redis-cli -n "$REDIS_DB" XINFO STREAM "$stream" 2>/dev/null \
        | awk '$1=="last-generated-id"{print $2; exit}' \
        | tr -d '"' \
        | tr -d '\r' \
        || true
}

_wait_stream_advance_by_id() {
    # Usage: _wait_stream_advance_by_id <stream> <seconds_total> <sleep_step>
    # Success if last-generated-id changes within the window.
    local stream="$1"
    local seconds_total="${2:-6}"
    local sleep_step="${3:-1}"

    local a b
    a="$(_last_stream_id "$stream")"
    a="${a:-}"

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
    # Usage: _wait_group_health <stream> <group> <seconds_total> <sleep_step>
    # Success if group exists and lag==0 and pending==0 at least once within window.
    local stream="$1"
    local group="$2"
    local seconds_total="${3:-6}"
    local sleep_step="${4:-1}"

    local elapsed=0
    while (( elapsed < seconds_total )); do
        local out pending lag
        out="$(redis-cli -n "$REDIS_DB" XINFO GROUPS "$stream" 2>/dev/null || true)"
        pending="$(printf "%s\n" "$out" \
            | awk -v g="$group" '
                $0 ~ "\"name\"" {getline; name=$0; gsub(/"/,"",name)}
                $0 ~ "\"pending\"" {getline; p=$0}
                $0 ~ "\"lag\"" {getline; l=$0; if(name==g){print p; exit}}
            ' | tr -d '\r' | head -n1)"
        lag="$(printf "%s\n" "$out" \
            | awk -v g="$group" '
                $0 ~ "\"name\"" {getline; name=$0; gsub(/"/,"",name)}
                $0 ~ "\"lag\"" {getline; l=$0; if(name==g){print l; exit}}
            ' | tr -d '\r' | head -n1)"

        # If group not found, pending/lag will be empty
        if [[ -n "${pending:-}" ]] && [[ -n "${lag:-}" ]]; then
            # pending/lag lines look like: (integer) 0
            if echo "$pending" | grep -q "(integer) 0" && echo "$lag" | grep -q "(integer) 0"; then
                return 0
            fi
        fi

        sleep "$sleep_step"
        elapsed=$((elapsed + sleep_step))
    done
    return 1
}

# -----------------------------
# Heal critical singleton pidfiles (prevents stale pid_dead alarms)
# -----------------------------
heal_pidfile "aggregator"      "orchestration/aggregator/run_aggregator.py"
heal_pidfile "top_selector"    "orchestration/selector/top_selector.py"
heal_pidfile "master_executor" "orchestration/executor/master_executor.py"
heal_pidfile "intent_bridge"   "orchestration/executor/intent_bridge.py"

# -----------------------------
# DRY_RUN reset policy (SAFER)
# -----------------------------
if is_truthy "${DRY_RUN:-1}"; then
    if ! any_orch_running; then
        echo "[START] DRY_RUN -> resetting open_positions_state"
        redis-cli -n "$REDIS_DB" DEL open_positions_state >/dev/null 2>&1 || true
    else
        echo "[SKIP] DRY_RUN reset (processes already running)"
    fi
fi

# -----------------------------
# Optional: "sterile" mode
# -----------------------------
# STERILE=1 -> sadece master_executor + intent_bridge başlatır
STERILE="${STERILE:-0}"

if is_truthy "$STERILE"; then
    echo "[MODE] STERILE=1 -> starting only master_executor + intent_bridge"

    # Heal again just in case (sterile runs standalone)
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
# SCANNERS: 24 symbols -> 8 workers (3 symbols each)
# Priority:
#   1) Wn_SYMBOLS override (if set)
#   2) SYMBOLS_24 auto-split
# -----------------------------
SYMBOLS_24="${SYMBOLS_24:-BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,MATICUSDT,DOTUSDT,LTCUSDT,TRXUSDT,ATOMUSDT,OPUSDT,ARBUSDT,APTUSDT,INJUSDT,SUIUSDT,FILUSDT,NEARUSDT,ETCUSDT,UNIUSDT,AAVEUSDT}"
WORKER_INTERVAL="${WORKER_INTERVAL:-5m}"
SCANNER_MODE="${SCANNER_MODE:-fast}"

# worker layer spam gates (env override allowed)
WORKER_MAX_SPREAD_PCT="${WORKER_MAX_SPREAD_PCT:-0.0010}"  # 0.10%
WORKER_MAX_ATR_PCT="${WORKER_MAX_ATR_PCT:-0.0300}"        # 3.0%
WORKER_MIN_CONF="${WORKER_MIN_CONF:-0.45}"

IFS=',' read -r -a symarr <<<"$SYMBOLS_24"

join3() {
  local i="$1"
  local a="${symarr[$i]:-}"
  local b="${symarr[$((i+1))]:-}"
  local c="${symarr[$((i+2))]:-}"
  echo "${a},${b},${c}"
}

pick_worker_symbols() {
  local n="$1"
  local var="W${n}_SYMBOLS"
  local override="${!var:-}"
  if [[ -n "${override}" ]]; then
    echo "${override}"
    return 0
  fi
  local idx=$(( (n-1) * 3 ))
  join3 "$idx"
}

start_one "scanner_w1" env WORKER_ID="w1" WORKER_SYMBOLS="$(pick_worker_symbols 1)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w2" env WORKER_ID="w2" WORKER_SYMBOLS="$(pick_worker_symbols 2)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w3" env WORKER_ID="w3" WORKER_SYMBOLS="$(pick_worker_symbols 3)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w4" env WORKER_ID="w4" WORKER_SYMBOLS="$(pick_worker_symbols 4)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w5" env WORKER_ID="w5" WORKER_SYMBOLS="$(pick_worker_symbols 5)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w6" env WORKER_ID="w6" WORKER_SYMBOLS="$(pick_worker_symbols 6)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w7" env WORKER_ID="w7" WORKER_SYMBOLS="$(pick_worker_symbols 7)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w8" env WORKER_ID="w8" WORKER_SYMBOLS="$(pick_worker_symbols 8)" WORKER_INTERVAL="$WORKER_INTERVAL" SCANNER_MODE="$SCANNER_MODE" WORKER_PUBLISH_HOLD=0 WORKER_MAX_SPREAD_PCT="$WORKER_MAX_SPREAD_PCT" WORKER_MAX_ATR_PCT="$WORKER_MAX_ATR_PCT" WORKER_MIN_CONF="$WORKER_MIN_CONF" "$PY" -u orchestration/scanners/worker_stub.py

# -----------------------------
# Downstream (singletons)
# -----------------------------
# Heal just before starting (extra safety if something changed during scanner starts)
heal_pidfile "aggregator"      "orchestration/aggregator/run_aggregator.py"
heal_pidfile "top_selector"    "orchestration/selector/top_selector.py"
heal_pidfile "master_executor" "orchestration/executor/master_executor.py"
heal_pidfile "intent_bridge"   "orchestration/executor/intent_bridge.py"

start_one "aggregator"      "$PY" -u orchestration/aggregator/run_aggregator.py
start_one "top_selector"    "$PY" -u orchestration/selector/top_selector.py
start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py
start_one "intent_bridge"   "$PY" -u orchestration/executor/intent_bridge.py

# -----------------------------
# Readiness (optional)
# -----------------------------
WAIT_READY="${WAIT_READY:-1}"
if is_truthy "$WAIT_READY"; then
    echo
    echo "=== Readiness check (best effort) ==="

    # 1) signals_stream: eski yöntem (len artışı) + yeni yöntem (ID advance)
    if wait_stream_growth "signals_stream" 1 6; then
        echo "[OK] signals_stream growing (len)"
    else
        if _wait_stream_advance_by_id "signals_stream" 6 1; then
            echo "[OK] signals_stream advanced (id)"
        else
            echo "[WARN] signals_stream did not advance in time (might still be OK)"
        fi
    fi

    # 2) trade_intents_stream: master -> bridge hattı
    if _wait_stream_advance_by_id "trade_intents_stream" 6 1; then
        echo "[OK] trade_intents_stream advanced (id)"
    else
        echo "[WARN] trade_intents_stream did not advance in time (might still be OK)"
    fi

    # 3) exec_events_stream: bridge output. MAXLEN=5000 olduğundan len sabit kalabilir,
    # bu yüzden id-advance kontrolü daha doğru.
    if _wait_stream_advance_by_id "exec_events_stream" 6 1; then
        echo "[OK] exec_events_stream advanced (id)"
    else
        echo "[WARN] exec_events_stream did not advance in time (might still be OK)"
    fi

    # 4) Optional: group health checks
    if _wait_group_health "trade_intents_stream" "$BRIDGE_GROUP" 6 1; then
        echo "[OK] trade_intents_stream group healthy (group=$BRIDGE_GROUP pending=0 lag=0)"
    else
        echo "[WARN] trade_intents_stream group not healthy/visible yet (group=$BRIDGE_GROUP)"
    fi

    if _wait_group_health "top5_stream" "$MASTER_GROUP" 6 1; then
        echo "[OK] top5_stream group healthy (group=$MASTER_GROUP pending=0 lag=0)"
    else
        echo "[WARN] top5_stream group not healthy/visible yet (group=$MASTER_GROUP)"
    fi
fi

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"
