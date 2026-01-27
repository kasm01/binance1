#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

D="${XDG_STATE_HOME:-$HOME/.local/state}/binance1/postmortem"
mkdir -p "$D"

# -----------------------------
# Timestamp (epoch + human)
# -----------------------------
TS_EPOCH="$(/usr/bin/python3 - <<'PY'
import time
print(int(time.time()))
PY
)"

TS_HUMAN="$(/usr/bin/python3 - <<'PY'
import time
print(time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
PY
)"

# Guard: if TS missing, do nothing (prevents *_ .txt and blank events)
if [[ -z "${TS_EPOCH:-}" || -z "${TS_HUMAN:-}" ]]; then
  exit 0
fi

# -----------------------------
# Write events + snapshots
# -----------------------------
/usr/bin/echo "[POSTMORTEM] ${TS_HUMAN} (${TS_EPOCH}) stop/restart snapshot" >> "$D/events.log" || true

/usr/bin/ps -eo pid,ppid,etimes,stat,cmd > "$D/ps_all_${TS_EPOCH}.txt" || true
/usr/bin/ps -eo pid,ppid,etimes,stat,cmd | /bin/egrep -i \
  "orchestration/(selector/top_selector|aggregator/run_aggregator|executor/master_executor|executor/intent_bridge|scanners/worker_stub)\.py" \
  > "$D/ps_orch_${TS_EPOCH}.txt" || true

for f in \
  "$HOME/binance1/logs/orch/top_selector.log" \
  "$HOME/binance1/logs/orch/master_executor.log" \
  "$HOME/binance1/logs/orch/intent_bridge.log" \
  "$HOME/binance1/logs/orch/aggregator.log"
do
  if [[ -f "$f" ]]; then
    /usr/bin/tail -n 200 "$f" > "$D/$(/usr/bin/basename "$f" .log)_tail_${TS_EPOCH}.txt" || true
  fi
done

/usr/bin/journalctl --user -u binance1-orch.service --since "10 min ago" --no-pager > "$D/journal_orch_${TS_EPOCH}.txt" || true
/usr/bin/journalctl --user -u binance1-orch-watchdog.service --since "10 min ago" --no-pager > "$D/journal_watchdog_${TS_EPOCH}.txt" || true
/usr/bin/journalctl --user -u binance1-orch-health.service --since "10 min ago" --no-pager > "$D/journal_health_${TS_EPOCH}.txt" || true

# -----------------------------
# Size cap ~200MB (best-effort)
# keep deleting oldest *.txt until under cap
# -----------------------------
MAX_KB=$((200*1024))
while [[ "$(/usr/bin/du -sk "$D" | /usr/bin/awk '{print $1}')" -gt "$MAX_KB" ]]; do
  /usr/bin/ls -1t "$D"/*.txt 2>/dev/null | /usr/bin/tail -n 1 | /usr/bin/xargs -r /usr/bin/rm -f
done

exit 0
