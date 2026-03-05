#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/binance1}"
SUP_DIR="$BASE_DIR/logs/supervisor"
PID_MAIN_FILE="$SUP_DIR/main.pid"
PID_BRIDGE_FILE="$SUP_DIR/bridge.pid"

echo "[supervisor_stop] begin (MAINPID=${MAINPID:-unset})"

# 1) Önce supervisor'ı kibar kapat (loop restart etmesin)
if [[ -n "${MAINPID:-}" ]] && kill -0 "$MAINPID" >/dev/null 2>&1; then
  echo "[supervisor_stop] sending SIGTERM to supervisor MAINPID=$MAINPID"
  kill -TERM "$MAINPID" >/dev/null 2>&1 || true

  # 45sn bekle (supervisor loop exit + kendi child cleanup'ı için)
  for _ in {1..45}; do
    kill -0 "$MAINPID" >/dev/null 2>&1 || break
    sleep 1
  done

  if kill -0 "$MAINPID" >/dev/null 2>&1; then
    echo "[supervisor_stop] supervisor still alive -> SIGKILL MAINPID=$MAINPID"
    kill -KILL "$MAINPID" >/dev/null 2>&1 || true
  fi
else
  echo "[supervisor_stop] MAINPID not running; continuing"
fi

# 2) Eğer arkada child kaldıysa (nadir), pidfile ile temizle
kill_pidfile() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local pid
  pid="$(cat "$f" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 0

  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "[supervisor_stop] stopping leftover pid=$pid (file=$f)"
    kill -TERM "$pid" >/dev/null 2>&1 || true
    for _ in {1..10}; do
      kill -0 "$pid" >/dev/null 2>&1 || break
      sleep 1
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "[supervisor_stop] force killing leftover pid=$pid"
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  fi

  : >"$f" 2>/dev/null || true
}

kill_pidfile "$PID_MAIN_FILE"
kill_pidfile "$PID_BRIDGE_FILE"

echo "[supervisor_stop] done"
