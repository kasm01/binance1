#!/usr/bin/env bash
set -euo pipefail

TS_EPOCH="$(/usr/bin/python3 -c 'import time; print(int(time.time()))')"
TS_HUMAN="$(/usr/bin/python3 -c 'import time; print(time.strftime("%Y%m%d_%H%M%S", time.gmtime()))')"

D="$HOME/.local/state/binance1/postmortem"
mkdir -p "$D"

# guard: TS empty => do nothing (prevents *_ .txt)
[ -n "${TS_EPOCH:-}" ] || exit 0

echo "[POSTMORTEM] ${TS_HUMAN} (${TS_EPOCH}) stop/restart snapshot" >> "$D/events.log"

ps -eo pid,ppid,etimes,stat,cmd > "$D/ps_all_${TS_EPOCH}.txt" || true
ps -eo pid,ppid,etimes,stat,cmd | egrep -i \
  'orchestration/(selector/top_selector|aggregator/run_aggregator|executor/master_executor|executor/intent_bridge|scanners/worker_stub)\.py' \
  > "$D/ps_orch_${TS_EPOCH}.txt" || true

for f in \
  "$HOME/binance1/logs/orch/top_selector.log" \
  "$HOME/binance1/logs/orch/master_executor.log" \
  "$HOME/binance1/logs/orch/intent_bridge.log" \
  "$HOME/binance1/logs/orch/aggregator.log"
do
  [ -f "$f" ] && tail -n 200 "$f" > "$D/$(basename "$f" .log)_tail_${TS_EPOCH}.txt"
done

journalctl --user -u binance1-orch.service --since "10 min ago" --no-pager > "$D/journal_orch_${TS_EPOCH}.txt" || true
journalctl --user -u binance1-orch-watchdog.service --since "10 min ago" --no-pager > "$D/journal_watchdog_${TS_EPOCH}.txt" || true
journalctl --user -u binance1-orch-health.service --since "10 min ago" --no-pager > "$D/journal_health_${TS_EPOCH}.txt" || true

# total size cap ~200MB (best-effort)
MAX_KB=$((200*1024))
while [[ $(du -sk "$D" | awk '{print $1}') -gt $MAX_KB ]]; do
  ls -1t "$D"/*.txt 2>/dev/null | tail -n 1 | xargs -r rm -f
done

exit 0
