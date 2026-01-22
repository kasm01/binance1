#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

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
    scanner) echo "orchestration/scanners/worker_stub.py" ;;
    aggregator) echo "Aggregator(RedisBus()).run_forever" ;;
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

  # pid boşsa temizle
  if [[ -z "${oldpid:-}" ]]; then
    rm -f "$pidfile"
    return 0
  fi

  # süreç yaşamıyorsa temizle
  if ! kill -0 "$oldpid" 2>/dev/null; then
    rm -f "$pidfile"
    return 0
  fi

  # yaşıyorsa ama komut uyuşmuyorsa temizle (yanlış pid reuse engeli)
  expect="$(expected_cmd "$name")"
  cmd="$(ps -p "$oldpid" -o args= 2>/dev/null || true)"

  if [[ -n "$expect" ]] && [[ -n "$cmd" ]] && echo "$cmd" | grep -Fq "$expect"; then
    echo "[SKIP] $name already running (pid=$oldpid)"
    return 1  # "already running" sinyali
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

  # varsa pidfile -> yaşıyor mu + komut match mi?
  if ! cleanup_stale_pidfile_if_needed "$name"; then
    # already running
    return 0
  fi

  echo "[START] $name -> $logfile"
  nohup env PYTHONPATH="$PWD" "$@" >>"$logfile" 2>&1 &
  echo $! > "$pidfile"

  # start check
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

# -----------------------------
# DRY_RUN reset policy (SAFER)
# -----------------------------
# Eskiden sadece open_positions_state temizleniyordu.
# Burada iki güvenlik iyileştirmesi:
#  1) Reset sadece "hiçbir orch prosesi çalışmıyorsa" yapılır.
#  2) Stream DEL yapmak istersen, bunu ancak "sterile mode" veya özel flag ile yap.
any_running=0
for f in "$RUNDIR"/*.pid; do
  [[ -e "$f" ]] || continue
  p="$(cat "$f" 2>/dev/null || true)"
  if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then
    any_running=1
    break
  fi
done

if is_truthy "${DRY_RUN:-1}"; then
  if [[ "$any_running" -eq 0 ]]; then
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
# Böylece scanner/aggregator/top_selector araya intent sokmaz.
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
# Full orch start
# -----------------------------
# 1) Scanner (stub)
start_one "scanner" "$PY" -u orchestration/scanners/worker_stub.py

# 2) Aggregator
start_one "aggregator" "$PY" -u -c \
  "from orchestration.event_bus.redis_bus import RedisBus; from orchestration.aggregator.aggregator import Aggregator; Aggregator(RedisBus()).run_forever()"

# 3) Top Selector
start_one "top_selector" "$PY" -u orchestration/selector/top_selector.py

# 4) Master Executor
start_one "master_executor" "$PY" -u orchestration/executor/master_executor.py

# 5) Intent Bridge
start_one "intent_bridge" "$PY" -u orchestration/executor/intent_bridge.py

echo
echo "All start commands issued."
echo "Tail logs: tail -n 200 -f logs/orch/*.log"

