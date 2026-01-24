#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# -----------------------------
# Single-instance lock
# -----------------------------
exec 9>/tmp/binance1_orch_watchdog.lock
flock -n 9 || exit 0

# -----------------------------
# Tuning
# -----------------------------
SLEEP_SEC="${WATCHDOG_SLEEP_SEC:-2}"
MAX_IDLE_SEC=""   # 2s pencerede artış yoksa bile son aktivite <=15s ise OK

need() { command -v "$1" >/dev/null 2>&1 || { echo "[WD][FAIL] missing: $1"; exit 2; }; }
need redis-cli
need pgrep
need awk
need date

# -----------------------------
# Process health (must exist)
# -----------------------------
must_patterns=(
  "orchestration/scanners/worker_stub.py"
  "orchestration/aggregator/run_aggregator.py"
  "orchestration/selector/top_selector.py"
  "orchestration/executor/master_executor.py"
  "orchestration/executor/intent_bridge.py"
)

for pat in "${must_patterns[@]}"; do
  if ! pgrep -af "$pat" >/dev/null 2>&1; then
    echo "[WD][FAIL] missing process: $pat -> restarting binance1-orch.service"
    systemctl --user restart binance1-orch.service
    exit 0
  fi
done

# -----------------------------
# Stream activity (OR logic)
#   - Prefer IDs (XLEN can go down with trimming)
# -----------------------------
last_id() {
  # prints stream last entry id or empty
  redis-cli XREVRANGE "$1" + - COUNT 1 2>/dev/null | head -n 1 | tr -d '\r' || true
}

id_ms() {
  # from "1769206846267-0" -> "1769206846267"
  awk -F'-' '{print $1}' <<<"${1:-}" 2>/dev/null || true
}

now_ms() {
  date +%s%3N
}

sig1="$(last_id signals_stream)"
exe1="$(last_id exec_events_stream)"

sleep "$SLEEP_SEC"

sig2="$(last_id signals_stream)"
exe2="$(last_id exec_events_stream)"

# If either stream moved -> OK (OR)
if [[ -n "${sig1:-}" && -n "${sig2:-}" && "$sig2" != "$sig1" ]] || [[ -n "${exe1:-}" && -n "${exe2:-}" && "$exe2" != "$exe1" ]]; then
  echo "[WD][OK] stream activity: signals ${sig1:-na} -> ${sig2:-na}, exec ${exe1:-na} -> ${exe2:-na}"
  exit 0
fi

# No movement in window: allow if recent activity exists (either stream)
nm="$(now_ms)"

sig2m="$(id_ms "$sig2")"
exe2m="$(id_ms "$exe2")"

age_sig="na"
age_exe="na"

if [[ -n "${sig2m:-}" ]]; then
  age_sig=$(( (nm - sig2m) / 1000 ))
fi
if [[ -n "${exe2m:-}" ]]; then
  age_exe=$(( (nm - exe2m) / 1000 ))
fi

recent_ok=0
if [[ "$age_sig" != "na" && "$age_sig" -le "$MAX_IDLE_SEC" ]]; then recent_ok=1; fi
if [[ "$age_exe" != "na" && "$age_exe" -le "$MAX_IDLE_SEC" ]]; then recent_ok=1; fi

if [[ "$recent_ok" -eq 1 ]]; then
  echo "[WD][OK] no growth in ${SLEEP_SEC}s but recent activity (age_sig=${age_sig}s age_exec=${age_exe}s)"
  exit 0
fi

echo "[WD][FAIL] no activity (>${MAX_IDLE_SEC}s) -> restarting binance1-orch.service (age_sig=${age_sig}s age_exec=${age_exe}s)"
systemctl --user restart binance1-orch.service
exit 0
