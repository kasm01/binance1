#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (override via env)
# -----------------------------
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"

IN_STREAM="${BRIDGE_IN_STREAM:-trade_intents_stream}"
GROUP="${BRIDGE_GROUP:-bridge_g}"
SERVICE="${ORCH_SERVICE:-binance1-orch.service}"

STATE_KEY="${BRIDGE_STATE_KEY:-open_positions_state}"
POS_PREFIX="${POSITION_KEY_PREFIX:-positions}"

SYMBOL="${TEST_SYMBOL:-DOGEUSDT}"
INTERVAL="${TEST_INTERVAL:-5m}"
OPEN_ID="${TEST_OPEN_ID:-manual-open-1}"
CLOSE_ID="${TEST_CLOSE_ID:-manual-close-1}"
OPEN_PRICE="${TEST_OPEN_PRICE:-0.1234}"
CLOSE_PRICE="${TEST_CLOSE_PRICE:-0.1240}"

SLEEP_AFTER_RESTART="${SLEEP_AFTER_RESTART:-1}"
SLEEP_AFTER_INTENT="${SLEEP_AFTER_INTENT:-1}"

redis_cmd() {
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" "$@"
}

say() { printf "\n\033[1m%s\033[0m\n" "$*"; }

require_bin() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing binary: $1" >&2; exit 1; }
}

# -----------------------------
# Preflight
# -----------------------------
require_bin redis-cli
require_bin jq
require_bin systemctl
require_bin pgrep
require_bin awk
require_bin egrep
require_bin tr
require_bin head
require_bin sleep
require_bin journalctl

say "[0] Preflight"
echo "REDIS=$REDIS_HOST:$REDIS_PORT/$REDIS_DB"
echo "SERVICE=$SERVICE"
echo "STREAM=$IN_STREAM GROUP=$GROUP"
echo "STATE_KEY=$STATE_KEY POS_PREFIX=$POS_PREFIX"
echo "TEST_SYMBOL=$SYMBOL interval=$INTERVAL open_price=$OPEN_PRICE close_price=$CLOSE_PRICE"

say "[1] Redis ping"
redis_cmd PING

# -----------------------------
# Restart service
# -----------------------------
say "[2] Restart service"
systemctl --user daemon-reload
systemctl --user restart "$SERVICE"
sleep "$SLEEP_AFTER_RESTART"

PID="$(pgrep -f "orchestration/executor/intent_bridge.py" | head -n1 | awk '{print $1}')"
if [[ -z "${PID:-}" ]]; then
  echo "ERROR: intent_bridge.py PID not found" >&2
  journalctl --user -u "$SERVICE" -n 200 --no-pager || true
  exit 1
fi
echo "PID=$PID"

say "[3] Runtime env (critical)"
tr '\0' '\n' < "/proc/$PID/environ" | egrep \
'DRY_RUN=|ARMED=|LIVE_KILL_SWITCH=|BRIDGE_DRYRUN_CALL_EXECUTOR=|BRIDGE_DRYRUN_WRITE_STATE=|BRIDGE_STATE_TTL_SEC=|BRIDGE_STATE_KEY=|REDIS_HOST=|REDIS_PORT=|REDIS_DB=|POSITION_REDIS_URL=|POSITION_REDIS_DB=|POSITION_KEY_PREFIX=' || true

# -----------------------------
# Reset keys
# -----------------------------
say "[4] Reset state keys (TEST ONLY)"
redis_cmd DEL "$STATE_KEY" "${POS_PREFIX}:${SYMBOL}" >/dev/null
echo "TTL($STATE_KEY) => $(redis_cmd TTL "$STATE_KEY" || true)"
echo "GET ${POS_PREFIX}:${SYMBOL} => $(redis_cmd GET "${POS_PREFIX}:${SYMBOL}" || true)"

# -----------------------------
# Stream/group health
# -----------------------------
say "[5] Stream/group health"
redis_cmd XINFO GROUPS "$IN_STREAM" || true
redis_cmd XPENDING "$IN_STREAM" "$GROUP" || true

# -----------------------------
# OPEN intent
# -----------------------------
say "[6] OPEN intent -> $SYMBOL"
OPEN_XID="$(redis_cmd XADD "$IN_STREAM" "*" json \
"{\"items\":[{\"symbol\":\"$SYMBOL\",\"side\":\"long\",\"interval\":\"$INTERVAL\",\"intent_id\":\"$OPEN_ID\",\"price\":$OPEN_PRICE,\"score\":0.9,\"trail_pct\":0.05,\"stall_ttl_sec\":7200}]}" \
)"
echo "XADD(open) => $OPEN_XID"
sleep "$SLEEP_AFTER_INTENT"

say "[7] After OPEN: exec_events_stream (last 20)"
redis_cmd --raw XREVRANGE exec_events_stream + - COUNT 20 \
| awk 'prev=="json"{print; prev=""; next} {prev=$0}' \
| jq -r '.ts_utc+"  "+.kind+"  "+(.symbol//"")+"  "+(.method//"")+"  "+(.why//"")' || true

say "[8] After OPEN: state + positions"
echo "TTL($STATE_KEY) => $(redis_cmd TTL "$STATE_KEY" || true)"
echo "STATE:"
redis_cmd GET "$STATE_KEY" | jq . || true

echo "POSITION:"
redis_cmd GET "${POS_PREFIX}:${SYMBOL}" | jq . || true

# hard checks (non-fatal but warns)
if [[ "$(redis_cmd GET "${POS_PREFIX}:${SYMBOL}" || true)" == "" || "$(redis_cmd GET "${POS_PREFIX}:${SYMBOL}" || true)" == "(nil)" ]]; then
  echo "WARN: ${POS_PREFIX}:${SYMBOL} is nil after OPEN (executor path may not be writing positions)" >&2
fi

# -----------------------------
# CLOSE intent
# -----------------------------
say "[9] CLOSE intent -> $SYMBOL"
CLOSE_XID="$(redis_cmd XADD "$IN_STREAM" "*" json \
"{\"items\":[{\"symbol\":\"$SYMBOL\",\"side\":\"close\",\"interval\":\"$INTERVAL\",\"intent_id\":\"$CLOSE_ID\",\"price\":$CLOSE_PRICE}]}" \
)"
echo "XADD(close) => $CLOSE_XID"
sleep "$SLEEP_AFTER_INTENT"

say "[10] After CLOSE: exec_events_stream (last 30)"
redis_cmd --raw XREVRANGE exec_events_stream + - COUNT 30 \
| awk 'prev=="json"{print; prev=""; next} {prev=$0}' \
| jq -r '.ts_utc+"  "+.kind+"  "+(.symbol//"")+"  "+(.method//"")+"  "+(.why//"")' || true

say "[11] After CLOSE: state + positions"
echo "TTL($STATE_KEY) => $(redis_cmd TTL "$STATE_KEY" || true)"
echo "STATE:"
redis_cmd GET "$STATE_KEY" | jq . || true

echo "POSITION RAW:"
redis_cmd GET "${POS_PREFIX}:${SYMBOL}" || true

if [[ "$(redis_cmd GET "${POS_PREFIX}:${SYMBOL}" || true)" != "" && "$(redis_cmd GET "${POS_PREFIX}:${SYMBOL}" || true)" != "(nil)" ]]; then
  echo "WARN: ${POS_PREFIX}:${SYMBOL} still exists after CLOSE (close path may not be clearing positions)" >&2
fi

# -----------------------------
# Summary
# -----------------------------
say "[12] Summary"
echo "- OPEN stream id : $OPEN_XID"
echo "- CLOSE stream id: $CLOSE_XID"
echo "- TTL($STATE_KEY): $(redis_cmd TTL "$STATE_KEY" || true)"
echo "- keys matching ${POS_PREFIX}:* (top 20):"
redis_cmd KEYS "${POS_PREFIX}:*" | head -n 20 || true

say "[DONE] Smoketest finished."
