#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUNDIR="run"

stop_one () {
  local name="$1"
  local pidfile="${RUNDIR}/${name}.pid"

  if [[ ! -f "$pidfile" ]]; then
    echo "[SKIP] $name pidfile missing"
    return 0
  fi

  local pid
  pid="$(cat "$pidfile")"

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

  rm -f "$pidfile"
}

# ters sırada kapatmak daha güvenli
stop_one "intent_bridge"
stop_one "master_executor"
stop_one "top_selector"
stop_one "aggregator"
stop_one "scanner"

echo "All stop commands issued."
