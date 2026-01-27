#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

D="${XDG_STATE_HOME:-$HOME/.local/state}/binance1/postmortem"
mkdir -p "$D"

# -----------------------------
# Timestamp (epoch + human)
# -----------------------------
TS_EPOCH="$(/usr/bin/date +%s)"
TS_HUMAN="$(/usr/bin/date -u +%Y%m%d_%H%M%S)"

# Guard: if TS missing, do nothing (prevents *_ .txt and blank events)
if [[ -z "${TS_EPOCH:-}" || -z "${TS_HUMAN:-}" ]]; then
  exit 0
fi

# -----------------------------
# Helpers
# -----------------------------
log_event() {
  local msg="[POSTMORTEM] ${TS_HUMAN} (${TS_EPOCH}) stop/restart snapshot"
  /usr/bin/echo "$msg" >> "$D/events.log" || true
  # Also to journald (best-effort)
  if command -v /usr/bin/systemd-cat >/dev/null 2>&1; then
    /usr/bin/echo "$msg" | /usr/bin/systemd-cat -t orch_postmortem || true
  fi
}

write_atomic() {
  # write_atomic <final_path> <command...>
  local out="$1"; shift
  local tmp="${out}.tmp"
  ( "$@" ) > "$tmp" 2>&1 || true
  /usr/bin/mv -f "$tmp" "$out" || true
}

# -----------------------------
# Write events + snapshots
# -----------------------------
log_event

# Service status snapshot (best-effort)
write_atomic "$D/systemd_status_orch_${TS_EPOCH}.txt" \
  /bin/bash -lc '/usr/bin/systemctl --user --no-pager --full status binance1-orch.service || true'

# Full process snapshot (sorted by longest runtime first)
write_atomic "$D/ps_all_${TS_EPOCH}.txt" \
  /usr/bin/ps -eo pid,ppid,etimes,stat,cmd --sort=-etimes

# Orch-related process snapshot
write_atomic "$D/ps_orch_${TS_EPOCH}.txt" \
  /bin/bash -lc '
    /usr/bin/ps -eo pid,ppid,etimes,stat,cmd --sort=-etimes | /bin/egrep -i \
      "orchestration/(selector/top_selector|aggregator/run_aggregator|executor/master_executor|executor/intent_bridge|scanners/worker_stub)\.py" \
      || true
  '

# Tail key logs (best-effort)
for f in \
  "$HOME/binance1/logs/orch/top_selector.log" \
  "$HOME/binance1/logs/orch/master_executor.log" \
  "$HOME/binance1/logs/orch/intent_bridge.log" \
  "$HOME/binance1/logs/orch/aggregator.log"
do
  if [[ -f "$f" ]]; then
    base="$(/usr/bin/basename "$f" .log)"
    write_atomic "$D/${base}_tail_${TS_EPOCH}.txt" \
      /usr/bin/tail -n 200 "$f"
  fi
done

# Redis stream snapshots (best-effort)
write_atomic "$D/redis_streams_${TS_EPOCH}.txt" \
  /bin/bash -lc '
    if ! command -v redis-cli >/dev/null 2>&1; then
      echo "redis-cli not found; skipping"
      exit 0
    fi

    HOST="${REDIS_HOST:-127.0.0.1}"
    PORT="${REDIS_PORT:-6379}"
    PASS="${REDIS_PASSWORD:-}"

    RC=(redis-cli -h "$HOST" -p "$PORT")
    [[ -n "$PASS" ]] && RC+=(-a "$PASS")

    SIG="${REDIS_SIGNALS_STREAM:-signals_stream}"
    EXE="${REDIS_EXEC_STREAM:-exec_events_stream}"

    echo "Using redis host=$HOST port=$PORT auth=$([[ -n "$PASS" ]] && echo yes || echo no)"
    echo

    for s in "$SIG" "$EXE"; do
      echo "=== stream: $s ==="
      "${RC[@]}" XINFO STREAM "$s" 2>&1 || true
      "${RC[@]}" XLEN "$s" 2>&1 || true
      echo
    done
  '
# Watchdog state snapshot (best-effort)
if [[ -f "$HOME/.local/state/binance1/orch_watchdog.state" ]]; then
  write_atomic "$D/watchdog_state_${TS_EPOCH}.txt" \
    /bin/bash -lc 'cat "$HOME/.local/state/binance1/orch_watchdog.state"'
fi

# Journals (last 10 minutes)
write_atomic "$D/journal_orch_${TS_EPOCH}.txt" \
  /usr/bin/journalctl --user -u binance1-orch.service --since "10 min ago" --no-pager

write_atomic "$D/journal_watchdog_${TS_EPOCH}.txt" \
  /usr/bin/journalctl --user -u binance1-orch-watchdog.service --since "10 min ago" --no-pager

write_atomic "$D/journal_health_${TS_EPOCH}.txt" \
  /usr/bin/journalctl --user -u binance1-orch-health.service --since "10 min ago" --no-pager

# -----------------------------
# Size cap ~200MB (best-effort)
# keep deleting oldest *.txt until under cap
# -----------------------------
MAX_KB=$((200*1024))

while true; do
  cur_kb="$(/usr/bin/du -sk "$D" 2>/dev/null | /usr/bin/awk '{print $1}' || /usr/bin/echo 0)"
  [[ "${cur_kb:-0}" -le "$MAX_KB" ]] && break

  oldest="$(
    /bin/bash -lc 'shopt -s nullglob; ls -1t "'"$D"'"/*.txt 2>/dev/null | tail -n 1'
  )"
  [[ -z "${oldest:-}" ]] && break
  /usr/bin/rm -f "$oldest" || true
done

exit 0
