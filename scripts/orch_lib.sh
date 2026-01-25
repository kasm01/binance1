#!/usr/bin/env bash
set -euo pipefail

# This file is meant to be sourced by other scripts.
# shellcheck disable=SC2034

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT_DIR/venv/bin/python}"
LOGDIR="${LOGDIR:-$ROOT_DIR/logs/orch}"
RUNDIR="${RUNDIR:-$ROOT_DIR/run}"

mkdir -p "$LOGDIR" "$RUNDIR"

# --------
# helpers
# --------
is_truthy() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

_ts() { date +"%Y-%m-%d %H:%M:%S"; }

logi() { echo "[$(_ts)] $*"; }
logw() { echo "[$(_ts)] [WARN] $*" >&2; }
loge() { echo "[$(_ts)] [ERR ] $*" >&2; }

# pidfile -> expected pattern (ps args içinde aranır)
expected_cmd() {
  local name="${1:-}"
  case "$name" in
    scanner_*)      echo "orchestration/scanners/worker_stub.py" ;;
    aggregator)     echo "orchestration/aggregator/run_aggregator.py" ;;
    top_selector)   echo "orchestration/selector/top_selector.py" ;;
    master_executor)echo "orchestration/executor/master_executor.py" ;;
    intent_bridge)  echo "orchestration/executor/intent_bridge.py" ;;
    *)              echo "" ;;
  esac
}

# Returns:
#   0 -> continue (pidfile cleaned or doesn't exist, safe to start)
#   1 -> already running (pidfile healed/valid), caller should NOT start a new one
heal_pidfile_if_needed() {
  local name="$1"
  local pidfile="${RUNDIR}/${name}.pid"

  [[ -f "$pidfile" ]] || return 0

  local pid expect cmd newpid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  expect="$(expected_cmd "$name")"

  # If pid missing/empty -> try heal by pgrep, else remove
  if [[ -z "${pid:-}" ]]; then
    if [[ -n "${expect:-}" ]]; then
      newpid="$(pgrep -f "$expect" | head -n1 || true)"
      if [[ -n "${newpid:-}" ]]; then
        echo "$newpid" > "$pidfile"
        logw "[HEAL] $name pidfile empty -> healed (new=${newpid})"
        echo "[SKIP] $name already running (pid=$newpid)"
        return 1
      fi
    fi
    rm -f "$pidfile" || true
    logw "[CLEAN] $name pidfile empty -> removed"
    return 0
  fi

  # If pid dead -> try heal by pgrep, else remove
  if ! kill -0 "$pid" 2>/dev/null; then
    if [[ -n "${expect:-}" ]]; then
      newpid="$(pgrep -f "$expect" | head -n1 || true)"
      if [[ -n "${newpid:-}" ]]; then
        echo "$newpid" > "$pidfile"
        logw "[HEAL] $name pidfile dead -> healed (old=${pid} new=${newpid})"
        echo "[SKIP] $name already running (pid=$newpid)"
        return 1
      fi
    fi
    rm -f "$pidfile" || true
    logw "[CLEAN] $name pidfile dead -> removed (old=${pid})"
    return 0
  fi

  # pid alive: verify command matches expected (if we have expected)
  if [[ -n "${expect:-}" ]]; then
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ -n "${cmd:-}" ]] && echo "$cmd" | grep -Fq "$expect"; then
      echo "[SKIP] $name already running (pid=$pid)"
      return 1
    fi

    # mismatch: try heal by pgrep to find correct process, else remove pidfile
    newpid="$(pgrep -f "$expect" | head -n1 || true)"
    if [[ -n "${newpid:-}" ]]; then
      echo "$newpid" > "$pidfile"
      logw "[HEAL] $name pidfile cmd mismatch -> healed (old=${pid} new=${newpid})"
      echo "[SKIP] $name already running (pid=$newpid)"
      return 1
    fi

    logw "[CLEAN] $name pidfile cmd mismatch -> removed (pid=$pid)"
    logw "        expected: $expect"
    logw "        actual:   ${cmd:-na}"
    rm -f "$pidfile" || true
    return 0
  fi

  # No expected pattern: if pid alive, accept as running
  echo "[SKIP] $name already running (pid=$pid)"
  return 1
}

start_one() {
  local name="$1"; shift
  local logfile="${LOGDIR}/${name}.log"
  local pidfile="${RUNDIR}/${name}.pid"

  # If already running (or healed), don't start a new one
  if ! heal_pidfile_if_needed "$name"; then
    return 0
  fi

  echo "[START] $name -> $logfile"
  nohup env PYTHONPATH="$ROOT_DIR" "$@" >>"$logfile" 2>&1 &
  local newpid="$!"
  echo "$newpid" > "$pidfile"

  # quick sanity check
  sleep 0.25
  if kill -0 "$newpid" 2>/dev/null; then
    echo "[OK] $name pid=$newpid"
    return 0
  fi

  echo "[FAIL] $name did not start. Check: $logfile"
  rm -f "$pidfile" || true
  return 1
}

any_orch_running() {
  shopt -s nullglob
  local f p
  for f in "$RUNDIR"/*.pid; do
    p="$(cat "$f" 2>/dev/null || true)"
    if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

# --------
# stop
# --------
stop_pid() {
  local name="$1"
  local pid="$2"

  [[ -n "${pid:-}" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    [[ "${QUIET_STOP:-0}" == "1" ]] || echo "[STOP] ${name} pid=${pid} not running"
    return 0
  fi

  [[ "${QUIET_STOP:-0}" == "1" ]] || echo "[STOP] ${name} pid=${pid} (TERM)"
  kill -TERM "$pid" 2>/dev/null || true

  local i
  for i in 1 2 3 4 5; do
    sleep 0.2
    if ! kill -0 "$pid" 2>/dev/null; then
      [[ "${QUIET_STOP:-0}" == "1" ]] || echo "[STOP] ${name} pid=${pid} stopped"
      return 0
    fi
  done

  [[ "${QUIET_STOP:-0}" == "1" ]] || echo "[STOP] ${name} pid=${pid} still alive (KILL)"
  kill -KILL "$pid" 2>/dev/null || true
  sleep 0.1
  return 0
}

# --------
# redis helpers
# --------
has_redis_cli() { command -v redis-cli >/dev/null 2>&1; }

redis_xlen() {
  local stream="$1"
  has_redis_cli || { echo "NA"; return 0; }
  redis-cli XLEN "$stream" 2>/dev/null | tr -d '\r' || echo "NA"
}

redis_last_event() {
  local stream="$1"
  local count="${2:-1}"
  has_redis_cli || return 0
  redis-cli XREVRANGE "$stream" + - COUNT "$count" 2>/dev/null || true
}

wait_stream_growth() {
  # Wait until stream length grows (best-effort readiness check).
  # usage: wait_stream_growth stream min_increase timeout_sec
  local stream="$1"
  local min_inc="${2:-1}"
  local timeout="${3:-5}"

  has_redis_cli || return 0

  local start_len cur_len t0
  start_len="$(redis-cli XLEN "$stream" 2>/dev/null | tr -d '\r' || echo "0")"
  t0="$(date +%s)"

  while true; do
    cur_len="$(redis-cli XLEN "$stream" 2>/dev/null | tr -d '\r' || echo "$start_len")"
    if [[ "$cur_len" =~ ^[0-9]+$ ]] && [[ "$start_len" =~ ^[0-9]+$ ]]; then
      if (( cur_len - start_len >= min_inc )); then
        return 0
      fi
    fi
    if (( $(date +%s) - t0 >= timeout )); then
      return 1
    fi
    sleep 0.3
  done
}
