#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/orch_lib.sh

# Optional: load env safely (avoids parse errors)
if [[ -f ".env" ]]; then
    ./scripts/load_env.sh .env >/dev/null 2>&1 || true
fi

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
# -----------------------------
WORKER_INTERVAL="${WORKER_INTERVAL:-5m}"

W1="${W1_SYMBOLS:-BTCUSDT,ETHUSDT,BNBUSDT}"
W2="${W2_SYMBOLS:-SOLUSDT,XRPUSDT,ADAUSDT}"
W3="${W3_SYMBOLS:-DOGEUSDT,AVAXUSDT,LINKUSDT}"
W4="${W4_SYMBOLS:-MATICUSDT,DOTUSDT,LTCUSDT}"
W5="${W5_SYMBOLS:-TRXUSDT,ATOMUSDT,OPUSDT}"
W6="${W6_SYMBOLS:-ARBUSDT,APTUSDT,INJUSDT}"
W7="${W7_SYMBOLS:-SUIUSDT,FILUSDT,NEARUSDT}"
W8="${W8_SYMBOLS:-ETCUSDT,UNIUSDT,AAVEUSDT}"

start_one "scanner_w1" env WORKER_ID="w1" WORKER_SYMBOLS="$W1" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w2" env WORKER_ID="w2" WORKER_SYMBOLS="$W2" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w3" env WORKER_ID="w3" WORKER_SYMBOLS="$W3" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w4" env WORKER_ID="w4" WORKER_SYMBOLS="$W4" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w5" env WORKER_ID="w5" WORKER_SYMBOLS="$W5" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w6" env WORKER_ID="w6" WORKER_SYMBOLS="$W6" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w7" env WORKER_ID="w7" WORKER_SYMBOLS="$W7" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py
start_one "scanner_w8" env WORKER_ID="w8" WORKER_SYMBOLS="$W8" WORKER_INTERVAL="$WORKER_INTERVAL" WORKER_PUBLISH_HOLD=0 "$PY" -u orchestration/scanners/worker_stub.py

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
