#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/binance1"
STATE_FILE="$STATE_DIR/orch_health.state"
mkdir -p "$STATE_DIR"

# ---- Tuning (env ile override edilebilir) ----
GRACE_SEC="${HEALTH_GRACE_SEC:-90}"          # service restart/start sonrası grace
FAIL_THRESH="${HEALTH_FAIL_THRESH:-3}"       # ardışık fail sayısı
COOLDOWN_SEC="${HEALTH_COOLDOWN_SEC:-300}"   # restartlar arası min süre

now="$(date +%s)"

# state load
FAIL_COUNT=0
LAST_RESTART=0
LAST_OK=0
if [[ -f "$STATE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_FILE" || true
fi

# helper: state save
save_state() {
  cat >"$STATE_FILE" <<EOF
FAIL_COUNT=${FAIL_COUNT}
LAST_RESTART=${LAST_RESTART}
LAST_OK=${LAST_OK}
EOF
}

# ---- Service age (grace) ----
# user unit: ActiveEnterTimestampMonotonic daha sağlam ama kolay yol:
# systemctl show ile ActiveEnterTimestamp (wallclock) alıp parse etmek zor.
# Basit yaklaşım: restart olduğunda LAST_RESTART state ile zaten tutuluyor.
if (( LAST_RESTART > 0 )) && (( now - LAST_RESTART < GRACE_SEC )); then
  echo "[HEALTH][OK] in grace period (${now}-${LAST_RESTART}<${GRACE_SEC})"
  save_state
  exit 0
fi

# ---- kritik pidfile'lar ----
need=(aggregator top_selector master_executor intent_bridge)

ok=1
for n in "${need[@]}"; do
  pidfile="run/${n}.pid"
  if [[ ! -f "$pidfile" ]]; then ok=0; continue; fi
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "${pid:-}" ]] || ! kill -0 "$pid" 2>/dev/null; then ok=0; fi
done

# ---- stream kontrolü (soft) ----
# burada hard fail yapmıyoruz, sadece bilgi amaçlı
x1="$(redis-cli XLEN signals_stream 2>/dev/null || echo 0)"
sleep 1
x2="$(redis-cli XLEN signals_stream 2>/dev/null || echo 0)"
# xlen trim ile azalabilir; o yüzden sadece log
if [[ "$x2" -gt "$x1" ]]; then
  :
fi

if [[ "$ok" -eq 1 ]]; then
  FAIL_COUNT=0
  LAST_OK="$now"
  echo "[HEALTH][OK] pids ok"
  save_state
  exit 0
fi

# unhealthy -> fail counter
FAIL_COUNT=$((FAIL_COUNT + 1))
echo "[HEALTH][WARN] unhealthy (fail_count=${FAIL_COUNT}/${FAIL_THRESH})"

# cooldown check
if (( LAST_RESTART > 0 )) && (( now - LAST_RESTART < COOLDOWN_SEC )); then
  echo "[HEALTH][WARN] cooldown active, skipping restart (${now}-${LAST_RESTART}<${COOLDOWN_SEC})"
  save_state
  exit 0
fi

if (( FAIL_COUNT < FAIL_THRESH )); then
  save_state
  exit 0
fi

echo "[HEALTH][FAIL] threshold reached -> restarting binance1-orch.service"
FAIL_COUNT=0
LAST_RESTART="$now"
save_state
systemctl --user restart binance1-orch.service
exit 0
