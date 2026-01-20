#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUNDIR="run"

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

stop_one () {
  local name="$1"
  local pidfile="${RUNDIR}/${name}.pid"

  if [[ ! -f "$pidfile" ]]; then
    echo "[SKIP] $name pidfile missing"
    return 0
  fi

  local pid expect cmd
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  expect="$(expected_cmd "$name")"

  if [[ -z "${pid:-}" ]]; then
    echo "[CLEAN] $name pidfile empty -> removing"
    rm -f "$pidfile"
    return 0
  fi

  # PID yasiyor mu?
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[CLEAN] $name pid=$pid not running -> removing pidfile"
    rm -f "$pidfile"
    return 0
  fi

  # Cmd eslesiyor mu? (eslesmiyorsa kill atma!)
  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ -n "$expect" ]] && [[ -n "$cmd" ]] && ! echo "$cmd" | grep -Fq "$expect"; then
    echo "[CLEAN] $name pidfile exists but cmd mismatch (pid=$pid) -> NOT killing, removing pidfile"
    echo "        expected: $expect"
    echo "        actual:   $cmd"
    rm -f "$pidfile"
    return 0
  fi

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

  rm -f "$pidfile"
}

# ters sırada kapatmak daha güvenli
stop_one "intent_bridge"
stop_one "master_executor"
stop_one "top_selector"
stop_one "aggregator"
stop_one "scanner"

echo "All stop commands issued."
