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
    rm -f "$pidfile" || true
    return 0
  fi

  if ! kill -0 "$oldpid" 2>/dev/null; then
    rm -f "$pidfile" || true
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
  rm -f "$pidfile" || true
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
  nohup env PYTHONPATH="$ROOT_DIR" "$@" >>"$logfile" 2>&1 &
  echo $! > "$pidfile"

  sleep 0.25
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[OK] $name pid=$pid"
  else
    echo "[FAIL] $name did not start. Check: $logfile"
    rm -f "$pidfile" || true
    return 1
  fi
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
