#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

REDIS_DB="${REDIS_DB:-0}"

TOPSEL_GROUP="${TOPSEL_GROUP:-topsel_g}"
MASTER_GROUP="${MASTER_GROUP:-master_exec_g}"
BRIDGE_GROUP="${BRIDGE_GROUP:-bridge_g}"

streams=(signals_stream candidates_stream top5_stream trade_intents_stream exec_events_stream trade_intents_dlq)

need() { command -v "$1" >/dev/null 2>&1; }
if ! need redis-cli; then
  echo "[HEALTH] redis-cli not found"
  exit 1
fi

echo "=== Redis stream lengths (db=$REDIS_DB) ==="
for s in "${streams[@]}"; do
  printf "%-20s " "$s"
  redis-cli -n "$REDIS_DB" XLEN "$s" 2>/dev/null || echo "(n/a)"
done

echo
echo "=== Stream XINFO (length + last-generated-id) ==="
for s in "${streams[@]}"; do
  echo "--- $s ---"
  out="$(redis-cli -n "$REDIS_DB" XINFO STREAM "$s" 2>/dev/null || true)"
  one="$(echo "$out" | tr '\n' ' ')"
  len="$(echo "$one" | sed -n 's/.*length[[:space:]]*"\{0,1\}\([0-9]\+\)"\{0,1\}.*/\1/p')"
  lastid="$(echo "$one" | sed -n 's/.*last-generated-id[[:space:]]*"\{0,1\}\([^" ]\+\)"\{0,1\}.*/\1/p')"
  echo "length: ${len:-"(n/a)"}"
  echo "last-generated-id: ${lastid:-"(n/a)"}"
done

echo
echo "=== Groups ==="
echo "--- candidates_stream ---"
redis-cli -n "$REDIS_DB" XINFO GROUPS candidates_stream 2>/dev/null || echo "(no groups / no stream)"
echo "--- top5_stream ---"
redis-cli -n "$REDIS_DB" XINFO GROUPS top5_stream 2>/dev/null || echo "(no groups / no stream)"
echo "--- trade_intents_stream ---"
redis-cli -n "$REDIS_DB" XINFO GROUPS trade_intents_stream 2>/dev/null || echo "(no groups / no stream)"

echo
echo "=== Pending counts ==="
echo "- candidates_stream / ${TOPSEL_GROUP}"
redis-cli -n "$REDIS_DB" XPENDING candidates_stream "${TOPSEL_GROUP}" 2>/dev/null || echo "(no group)"
echo "- top5_stream / ${MASTER_GROUP}"
redis-cli -n "$REDIS_DB" XPENDING top5_stream "${MASTER_GROUP}" 2>/dev/null || echo "(no group)"
echo "- trade_intents_stream / ${BRIDGE_GROUP}"
redis-cli -n "$REDIS_DB" XPENDING trade_intents_stream "${BRIDGE_GROUP}" 2>/dev/null || echo "(no group)"

echo
echo "=== Open positions state ==="
redis-cli -n "$REDIS_DB" GET open_positions_state 2>/dev/null || true

echo
echo "=== Exec events (last 10) ==="
redis-cli -n "$REDIS_DB" XREVRANGE exec_events_stream + - COUNT 10 2>/dev/null || true

echo
echo "=== DLQ (last 10) ==="
redis-cli -n "$REDIS_DB" XREVRANGE trade_intents_dlq + - COUNT 10 2>/dev/null || true
