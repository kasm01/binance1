#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./venv/bin/python"
LOGDIR="logs/orch"
RUNDIR="run"

mkdir -p "$LOGDIR" "$RUNDIR"

start_one () {
  local name="$1"; shift
  local logfile="${LOGDIR}/${name}.log"
  local pidfile="${RUNDIR}/${name}.pid"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[SKIP] $name already running (pid=$(cat "$pidfile"))"
    return 0
  fi

  echo "[START] $name -> $logfile"
  nohup env PYTHONPATH="$PWD" "$@" >>"$logfile" 2>&1 &
  echo $! > "$pidfile"
  sleep 0.2
  if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[OK] $name pid=$(cat "$pidfile")"
  else
    echo "[FAIL] $name did not start. Check: $logfile"
    rm -f "$pidfile"
    return 1
  fi
}

# 1) Scanner (stub)
start_one "scanner" "$PY" -u orchestration/scanners/worker_stub.py

# 2) Aggregator (your -c runner)
start_one "aggregator" "$PY" -u -c "from orchestration.event_bus.redis_bus import RedisBus; from orchestration.aggregator.aggregator import Aggregator; Aggregator(RedisBus()).run_forever()"

# 3) Top Selector
# group tabanliya gecince TOPSEL_START_ID kullanmayacaksin; burada dokunmuyoruz.
start_one "top_selector" "$PY" -u orchestration/selector/top_selector.py

# 4) Master Executor
start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py

# 5) Intent Bridge
start_one "intent_bridge" "$PY" -u orchestration/executor/intent_bridge.py

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"
