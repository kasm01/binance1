#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUNDIR="run"

# pidfile -> expected pattern map
# (patternler ps args içinde aranır)
declare -A EXPECTED
EXPECTED["aggregator"]="orchestration/aggregator/run_aggregator.py"
EXPECTED["top_selector"]="orchestration/selector/top_selector.py"
EXPECTED["master_executor"]="orchestration/executor/master_executor.py"
EXPECTED["intent_bridge"]="orchestration/executor/intent_bridge.py"
# scanner_w1..scanner_wN dinamik -> altta otomatik bakılacak

cleanup_pidfiles () {
  shopt -s nullglob
  for pidfile in "${RUNDIR}"/*.pid; do
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile" 2>/dev/null || true)"

    # pid boşsa temizle
    if [[ -z "${pid}" ]]; then
      echo "[CLEAN] $name pidfile empty -> removing $pidfile"
      rm -f "$pidfile"
      continue
    fi

    # proses yoksa temizle
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[CLEAN] $name pid=$pid not running -> removing $pidfile"
      rm -f "$pidfile"
      continue
    fi

    # beklenen pattern belirle
    expect="${EXPECTED[$name]:-}"

    # scanner_w* için otomatik expected
    if [[ -z "$expect" && "$name" =~ ^scanner_w[0-9]+$ ]]; then
      expect="orchestration/scanners/worker_stub.py"
    fi

    # cmd mismatch kontrolü (beklenen pattern varsa)
    if [[ -n "$expect" ]]; then
      cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      if [[ -z "$cmd" ]]; then
        echo "[CLEAN] $name pid=$pid no cmd -> removing $pidfile"
        rm -f "$pidfile"
        continue
      fi
      if ! echo "$cmd" | grep -Fq "$expect"; then
        echo "[CLEAN] $name pid=$pid cmd mismatch"
        echo "        expected: $expect"
        echo "        actual:   $cmd"
        echo "        -> removing $pidfile (NOT killing process)"
        rm -f "$pidfile"
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
    echo "--- $f (tail -n 3) ---"
    tail -n 3 "$f" || true
  else
    echo "--- $f missing ---"
  fi
done
shopt -u nullglob
