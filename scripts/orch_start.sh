#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/orch_lib.sh

# Optional: load env safely (avoids parse errors)
if [[ -f ".env" ]]; then
  ./scripts/load_env.sh .env >/dev/null 2>&1 || true
fi

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
# Downstream
# -----------------------------
start_one "aggregator" "$PY" -u orchestration/aggregator/run_aggregator.py
start_one "top_selector" "$PY" -u orchestration/selector/top_selector.py
start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py
start_one "intent_bridge" "$PY" -u orchestration/executor/intent_bridge.py

# -----------------------------
# Readiness (optional)
# -----------------------------
WAIT_READY="${WAIT_READY:-1}"
if is_truthy "$WAIT_READY"; then
  echo
  echo "=== Readiness check (best effort) ==="
  if wait_stream_growth "signals_stream" 1 6; then
    echo "[OK] signals_stream growing"
  else
    echo "[WARN] signals_stream did not grow in time (might still be OK)"
  fi
  if wait_stream_growth "exec_events_stream" 1 6; then
    echo "[OK] exec_events_stream growing"
  else
    echo "[WARN] exec_events_stream did not grow in time (might still be OK)"
  fi
fi

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"
