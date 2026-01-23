#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Optional: load env safely (avoids "KEY: value" parse errors)
if [[ -f ".env" ]]; then
  ./scripts/load_env.sh .env >/dev/null 2>&1 || true
fi

PY="./venv/bin/python"
LOGDIR="logs/orch"
RUNDIR="run"

mkdir -p "$LOGDIR" "$RUNDIR"

# -----------------------------
# Helpers
# -----------------------------
is_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

# pidfile -> expected pattern (ps args içinde aranır)
expected_cmd() {
    local name="$1"
    case "$name" in
        scanner_*) echo "orchestration/scanners/worker_stub.py" ;;
        aggregator) echo "orchestration/aggregator/run_aggregator.py" ;;
        top_selector) echo "orchestration/selector/top_selector.py" ;;
        master_executor) echo "orchestration/executor/master_executor.py" ;;
        intent_bridge) echo "orchestration/executor/intent_bridge.py" ;;
        *) echo "" ;;
    esac
}

cleanup_stale_pidfile_if_needed() {
    local name="$1"
    local pidfile="${RUNDIR}/${name}.pid"

    [[ -f "$pidfile" ]] || return 0

    local oldpid cmd expect
    oldpid="$(cat "$pidfile" 2>/dev/null || true)"

    if [[ -z "${oldpid:-}" ]]; then
        rm -f "$pidfile"
        return 0
    fi

    if ! kill -0 "$oldpid" 2>/dev/null; then
        rm -f "$pidfile"
        return 0
    fi

    expect="$(expected_cmd "$name")"
    cmd="$(ps -p "$oldpid" -o args= 2>/dev/null || true)"

    if [[ -n "$expect" ]] && [[ -n "$cmd" ]] && echo "$cmd" | grep -Fq "$expect"; then
        echo "[SKIP] $name already running (pid=$oldpid)"
        return 1
    fi

    echo "[CLEAN] $name pidfile exists but cmd mismatch (pid=$oldpid) -> removing pidfile"
    echo "        expected: $expect"
    echo "        actual:   $cmd"
    rm -f "$pidfile"
    return 0
}

start_one() {
    local name="$1"; shift
    local logfile="${LOGDIR}/${name}.log"
    local pidfile="${RUNDIR}/${name}.pid"

    if ! cleanup_stale_pidfile_if_needed "$name"; then
        return 0
    fi

    echo "[START] $name -> $logfile"
    nohup env PYTHONPATH="$PWD" "$@" >>"$logfile" 2>&1 &
    echo $! > "$pidfile"

    sleep 0.25
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[OK] $name pid=$pid"
    else
        echo "[FAIL] $name did not start. Check: $logfile"
        rm -f "$pidfile"
        return 1
    fi
}

any_orch_running() {
    local f p
    for f in "$RUNDIR"/*.pid; do
        [[ -e "$f" ]] || continue
        p="$(cat "$f" 2>/dev/null || true)"
        if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

# -----------------------------
# DRY_RUN reset policy (SAFER)
# -----------------------------
if is_truthy "${DRY_RUN:-1}"; then
    if ! any_orch_running; then
        echo "[START] DRY_RUN -> resetting open_positions_state"
        redis-cli DEL open_positions_state >/dev/null 2>&1 || true
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
# Not:
# - WORKER_SYMBOLS listesini kendi 24 paritenle değiştir.
# - Her worker ayrı log + pidfile alır: scanner_w1.log / scanner_w1.pid vb.

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
# Full orch start (downstream)
# -----------------------------
start_one "aggregator" "$PY" -u orchestration/aggregator/run_aggregator.py

start_one "top_selector" "$PY" -u orchestration/selector/top_selector.py
start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py
start_one "intent_bridge" "$PY" -u orchestration/executor/intent_bridge.py

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"
