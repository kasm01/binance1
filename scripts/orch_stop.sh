#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUNDIR="run"

# name -> process match pattern (daha güvenli kapatma için)
pattern_for () {
  local name="$1"
  case "$name" in
    scanner) echo "orchestration/scanners/worker_stub.py" ;;
    aggregator) echo "Aggregator\\(RedisBus\\(\\)\\)\\.run_forever\\(\\)|orchestration/aggregator/aggregator.py" ;;
    top_selector) echo "orchestration/selector/top_selector.py" ;;
    master_executor) echo "orchestration/executor/master_executor.py" ;;
    intent_bridge) echo "orchestration/executor/intent_bridge.py" ;;
    *) echo "$name" ;;
  esac
}

stop_pid () {
  local name="$1"
  local pid="$2"

  if [[ -z "${pid:-}" ]]; then
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[STOP] $name pid=$pid"
    kill "$pid" 2>/dev/null || true

    # grace
    for i in {1..20}; do
      if kill -0 "$pid" 2>/dev/null; then
        sleep 0.2
      else
        break
      fi
    done

    if kill -0 "$pid" 2>/dev/null; then
      echo "[KILL] $name pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  else
    echo "[CLEAN] $name pid=$pid not running"
  fi
}

stop_one () {
  local name="$1"
  local pidfile="${RUNDIR}/${name}.pid"
  local pid=""

  # 1) önce pidfile'dan dene
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    stop_pid "$name" "$pid"
    rm -f "$pidfile"
    return 0
  fi

  # 2) pidfile yoksa: pgrep ile yakala (fail-safe)
  local pat
  pat="$(pattern_for "$name")"
  local pids
  pids="$(pgrep -af "$pat" | awk '{print $1}' | tr '\n' ' ' || true)"

  if [[ -z "${pids// }" ]]; then
    echo "[SKIP] $name pidfile missing and process not found"
    return 0
  fi

  echo "[STOP] $name pidfile missing, found pids: $pids"
  for p in $pids; do
    stop_pid "$name" "$p"
  done
}

# ters sırada kapatmak daha güvenli
stop_one "intent_bridge"
stop_one "master_executor"
stop_one "top_selector"
stop_one "aggregator"
stop_one "scanner"

echo "All stop commands issued."
