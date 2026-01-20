#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# group names (override edilebilir)
TOPSEL_GROUP="${TOPSEL_GROUP:-topsel_g}"
MASTER_GROUP="${MASTER_GROUP:-master_exec_g}"
BRIDGE_GROUP="${BRIDGE_GROUP:-bridge_g}"

streams=(signals_stream candidates_stream top5_stream trade_intents_stream exec_events_stream)

echo "=== Redis stream lengths ==="
for s in "${streams[@]}"; do
  printf "%-20s " "$s"
  redis-cli XLEN "$s" || true
done

echo
echo "=== Stream XINFO (length + last-generated-id) ==="
for s in "${streams[@]}"; do
  echo "--- $s ---"
  # XINFO STREAM çoğu sürümde "1) key 2) value ..." döner.
  # Burada tüm output'u tek satıra indirip regex ile çekiyoruz.
  out="$(redis-cli XINFO STREAM "$s" 2>/dev/null || true)"
  one_line="$(echo "$out" | tr '\n' ' ' )"

  # length
  len="$(echo "$one_line" | sed -n 's/.*length[[:space:]]*"\{0,1\}\([0-9]\+\)"\{0,1\}.*/\1/p')"
  # last-generated-id
  lastid="$(echo "$one_line" | sed -n 's/.*last-generated-id[[:space:]]*"\{0,1\}\([^" ]\+\)"\{0,1\}.*/\1/p')"

  echo "length: ${len:-"(n/a)"}"
  echo "last-generated-id: ${lastid:-"(n/a)"}"
done

echo
echo "=== Groups ==="
echo "--- candidates_stream ---"
redis-cli XINFO GROUPS candidates_stream 2>/dev/null || echo "(no groups / no stream)"
echo "--- top5_stream ---"
redis-cli XINFO GROUPS top5_stream 2>/dev/null || echo "(no groups / no stream)"
echo "--- trade_intents_stream ---"
redis-cli XINFO GROUPS trade_intents_stream 2>/dev/null || echo "(no groups / no stream)"

echo
echo "=== Pending counts (if groups exist) ==="
echo "- candidates_stream / ${TOPSEL_GROUP}"
redis-cli XPENDING candidates_stream "${TOPSEL_GROUP}" 2>/dev/null || echo "(no group)"
echo "- top5_stream / ${MASTER_GROUP}"
redis-cli XPENDING top5_stream "${MASTER_GROUP}" 2>/dev/null || echo "(no group)"
echo "- trade_intents_stream / ${BRIDGE_GROUP}"
redis-cli XPENDING trade_intents_stream "${BRIDGE_GROUP}" 2>/dev/null || echo "(no group)"

echo
echo "=== Open positions state ==="
redis-cli GET open_positions_state 2>/dev/null || true

echo
echo "=== Exec events (last 10) ==="
redis-cli XREVRANGE exec_events_stream + - COUNT 10 2>/dev/null || true

