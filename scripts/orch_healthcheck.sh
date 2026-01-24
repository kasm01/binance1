#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# kritik pidfile'lar
need=(aggregator top_selector master_executor intent_bridge)

ok=1
for n in "${need[@]}"; do
  pidfile="run/${n}.pid"
  if [[ ! -f "$pidfile" ]]; then ok=0; continue; fi
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "${pid:-}" ]] || ! kill -0 "$pid" 2>/dev/null; then ok=0; fi
done

# streamler büyüyor mu? (çok sıkı tutmuyoruz)
x1="$(redis-cli XLEN signals_stream 2>/dev/null || echo 0)"
sleep 1
x2="$(redis-cli XLEN signals_stream 2>/dev/null || echo 0)"
if [[ "$x2" -le "$x1" ]]; then
  # stub publish hold vb. olabilir; sadece soft-fail
  :
fi

if [[ "$ok" -eq 1 ]]; then
  exit 0
fi

echo "[HEALTH] orch unhealthy -> restarting user service"
systemctl --user restart binance1-orch.service
