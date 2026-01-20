#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Redis stream lengths ==="
for s in signals_stream candidates_stream top5_stream trade_intents_stream exec_events_stream; do
  printf "%-20s " "$s"
  redis-cli XLEN "$s"
done

echo
echo "=== Stream XINFO (length + last-generated-id) ==="
for s in signals_stream candidates_stream top5_stream trade_intents_stream exec_events_stream; do
  echo "--- $s ---"
  # XINFO STREAM çıktısı key/value gidiyor; length ve last-generated-id’yi birlikte bas
  redis-cli XINFO STREAM "$s" | awk '
    $1=="length" {print "length: " $2}
    $1=="last-generated-id" {print "last-generated-id: " $2}
  ' || true
done

echo
echo "=== Groups: candidates_stream ==="
redis-cli XINFO GROUPS candidates_stream || true
echo "=== Groups: top5_stream ==="
redis-cli XINFO GROUPS top5_stream || true
echo "=== Groups: trade_intents_stream ==="
redis-cli XINFO GROUPS trade_intents_stream || true

echo
echo "=== Pending counts (if groups exist) ==="
echo "- candidates_stream / topsel_g"
redis-cli XPENDING candidates_stream topsel_g 2>/dev/null || true
echo "- top5_stream / master_exec_g"
redis-cli XPENDING top5_stream master_exec_g 2>/dev/null || true
echo "- trade_intents_stream / bridge_g"
redis-cli XPENDING trade_intents_stream bridge_g 2>/dev/null || true

echo
echo "=== Open positions state ==="
redis-cli GET open_positions_state || true

echo
echo "=== Exec events (last 5) ==="
redis-cli XREVRANGE exec_events_stream + - COUNT 5 || true
