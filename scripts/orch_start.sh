#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./venv/bin/python"
LOGDIR="logs/orch"
RUNDIR="run"

mkdir -p "$LOGDIR" "$RUNDIR"

# DRY_RUN reset'i sadece 1 kez yap
if [ "${DRY_RUN:-1}" = "1" ] || [ "${DRY_RUN:-1}" = "true" ]; then
  echo "[START] DRY_RUN -> resetting open_positions_state"
  redis-cli DEL open_positions_state >/dev/null 2>&1 || true
fi

# pidfile -> expected pattern (ps args icinde aranir)
expected_cmd () {
  local name="$1"
  case "$name" in
    scanner) echo "orchestration/scanners/worker_stub.py" ;;
    aggregator) echo "Aggregator(RedisBus()).run_forever" ;;
    top_selector) echo "orchestration/selector/top_selector.py" ;;
    master_executor) echo "orchestration/executor/master_executor.py" ;;
    intent_bridge) echo "orchestration/executor/intent_bridge.py" ;;
    *) echo "" ;;
  esac
}

start_one () {
  local name="$1"; shift
  local logfile="${LOGDIR}/${name}.log"
  local pidfile="${RUNDIR}/${name}.pid"

  # pidfile varsa kontrol et (yasiyor mu + komut eslesiyor mu)
  if [[ -f "$pidfile" ]]; then
    local oldpid cmd expect
    oldpid="$(cat "$pidfile" 2>/dev/null || true)"

    if [[ -n "${oldpid:-}" ]] && kill -0 "$oldpid" 2>/dev/null; then
      expect="$(expected_cmd "$name")"
      cmd="$(ps -p "$oldpid" -o args= 2>/dev/null || true)"

      if [[ -n "$expect" ]] && [[ -n "$cmd" ]] && echo "$cmd" | grep -Fq "$expect"; then
        echo "[SKIP] $name already running (pid=$oldpid)"
        return 0
      else
        echo "[CLEAN] $name pidfile exists but cmd mismatch (pid=$oldpid) -> removing pidfile"
        echo "        expected: $expect"
        echo "        actual:   $cmd"
        rm -f "$pidfile"
      fi
    else
      rm -f "$pidfile"
    fi
  fi

  echo "[START] $name -> $logfile"
  nohup env PYTHONPATH="$PWD" "$@" >>"$logfile" 2>&1 &
  echo $! > "$pidfile"

  # start check
  sleep 0.2
  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[OK] $name pid=$pid"
  else
    echo "[FAIL] $name did not start. Check: $logfile"
    rm -f "$pidfile"
    return 1
  fi
}

# 1) Scanner (stub)
start_one "scanner" "$PY" -u orchestration/scanners/worker_stub.py

# 2) Aggregator
start_one "aggregator" "$PY" -u -c "from orchestration.event_bus.redis_bus import RedisBus; from orchestration.aggregator.aggregator import Aggregator; Aggregator(RedisBus()).run_forever()"

# 3) Top Selector
start_one "top_selector" "$PY" -u orchestration/selector/top_selector.py

# 4) Master Executor
start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py

# 5) Intent Bridge
start_one "intent_bridge" "$PY" -u orchestration/executor/intent_bridge.py

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"

