#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ./scripts/orch_lib.sh

STATUS_SHORT="${STATUS_SHORT:-0}"   # 1 -> Redis bölümünü basma
LOG_TAIL_N="${LOG_TAIL_N:-3}"

# pidfile -> expected pattern map
declare -A EXPECTED
EXPECTED["aggregator"]="orchestration/aggregator/run_aggregator.py"
EXPECTED["top_selector"]="orchestration/selector/top_selector.py"
EXPECTED["master_executor"]="orchestration/executor/master_executor.py"
EXPECTED["intent_bridge"]="orchestration/executor/intent_bridge.py"
# scanner_w* -> otomatik

cleanup_pidfiles () {
  shopt -s nullglob
  local pidfile name pid expect cmd
  for pidfile in "${RUNDIR}"/*.pid; do
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile" 2>/dev/null || true)"

    if [[ -z "${pid}" ]]; then
      echo "[CLEAN] $name pidfile empty -> removing $pidfile"
      rm -f "$pidfile" || true
      continue
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[CLEAN] $name pid=$pid not running -> removing $pidfile"
      rm -f "$pidfile" || true
      continue
    fi

    expect="${EXPECTED[$name]:-}"
    if [[ -z "$expect" && "$name" =~ ^scanner_w[0-9]+$ ]]; then
      expect="orchestration/scanners/worker_stub.py"
    fi

    if [[ -n "$expect" ]]; then
      cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      if [[ -z "$cmd" ]]; then
        echo "[CLEAN] $name pid=$pid no cmd -> removing $pidfile"
        rm -f "$pidfile" || true
        continue
      fi
      if ! echo "$cmd" | grep -Fq "$expect"; then
        echo "[CLEAN] $name pid=$pid cmd mismatch"
        echo "        expected: $expect"
        echo "        actual:   $cmd"
        echo "        -> removing $pidfile (NOT killing process)"
        rm -f "$pidfile" || true
        continue
      fi
    fi
  done
  shopt -u nullglob
}

cleanup_pidfiles

echo "=== PID files ==="
ls -1 "${RUNDIR}"/*.pid 2>/dev/null || echo "(none)"

echo
echo "=== Running orchestration processes (ps) ==="
ps aux | egrep \
  "orchestration/(scanners/worker_stub\.py|aggregator/run_aggregator\.py|selector/top_selector\.py|executor/master_executor\.py|executor/intent_bridge\.py)" \
  | grep -v grep || true

echo
echo "=== pgrep -af (orch) ==="
pgrep -af \
  "orchestration/(scanners/worker_stub\.py|aggregator/run_aggregator\.py|selector/top_selector\.py|executor/master_executor\.py|executor/intent_bridge\.py)" \
  || true

echo
echo "=== pgrep -af (aggregator) ==="
pgrep -af "orchestration/aggregator/run_aggregator\.py" || true

echo
echo "=== Quick log liveness (last lines) ==="
shopt -s nullglob
for f in logs/orch/scanner_w*.log logs/orch/aggregator.log logs/orch/top_selector.log logs/orch/master_executor.log logs/orch/intent_bridge.log; do
  if [[ -f "$f" ]]; then
    echo "--- $f (tail -n ${LOG_TAIL_N}) ---"
    tail -n "$LOG_TAIL_N" "$f" || true
  else
    echo "--- $f missing ---"
  fi
done
shopt -u nullglob

if is_truthy "$STATUS_SHORT"; then
  exit 0
fi

echo
echo "=== Redis stream health ==="
if ! has_redis_cli; then
  echo "(redis-cli not found)"
  exit 0
fi

printf "%-16s %s\n" "signals_stream"      "$(redis_xlen signals_stream)"
printf "%-16s %s\n" "candidates_stream"   "$(redis_xlen candidates_stream)"
printf "%-16s %s\n" "top5_stream"         "$(redis_xlen top5_stream)"
printf "%-16s %s\n" "trade_intents_stream" "$(redis_xlen trade_intents_stream)"
printf "%-16s %s\n" "exec_events_stream"  "$(redis_xlen exec_events_stream)"

echo
echo "--- last exec_events_stream (COUNT 1) ---"
redis_last_event exec_events_stream 1
