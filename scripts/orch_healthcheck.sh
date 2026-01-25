#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/binance1"
STATE_FILE="$STATE_DIR/orch_health.state"
mkdir -p "$STATE_DIR"

# ---- Tuning (env ile override) ----
GRACE_SEC="${HEALTH_GRACE_SEC:-90}"          # restart/start sonrası grace
FAIL_THRESH="${HEALTH_FAIL_THRESH:-3}"       # ardışık fail sayısı
COOLDOWN_SEC="${HEALTH_COOLDOWN_SEC:-300}"   # restartlar arası min süre

need() { command -v "$1" >/dev/null 2>&1; }

now="$(date +%s)"

# ---- state load ----
FAIL_COUNT=0
LAST_RESTART=0
LAST_OK=0
if [[ -f "$STATE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_FILE" || true
fi
FAIL_COUNT="${FAIL_COUNT:-0}"
LAST_RESTART="${LAST_RESTART:-0}"
LAST_OK="${LAST_OK:-0}"

# ---- helper: state save (atomic) ----
save_state() {
  local tmp="${STATE_FILE}.tmp"
  cat >"$tmp" <<EOF
FAIL_COUNT=${FAIL_COUNT}
LAST_RESTART=${LAST_RESTART}
LAST_OK=${LAST_OK}
EOF
  mv -f "$tmp" "$STATE_FILE"
}

# ---- grace ----
if (( LAST_RESTART > 0 )) && (( now - LAST_RESTART < GRACE_SEC )); then
  echo "[HEALTH][OK] in grace period (age=$((now - LAST_RESTART))s < ${GRACE_SEC}s)"
  save_state
  exit 0
fi

# ---- kritik pidfile'lar ----
need_names=(aggregator top_selector master_executor intent_bridge)

ok=1
reasons=()

for n in "${need_names[@]}"; do
  pidfile="run/${n}.pid"
  if [[ ! -f "$pidfile" ]]; then
    ok=0
    reasons+=("${n}:pidfile_missing")
    continue
  fi

  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "${pid:-}" ]]; then
    ok=0
    reasons+=("${n}:pid_empty")
    continue
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    ok=0
    reasons+=("${n}:pid_dead(${pid})")
    continue
  fi
done

# ---- stream kontrolü (soft, bilgi amaçlı) ----
# redis-cli yoksa health'i sadece pidlerle değerlendir
if need redis-cli; then
  x1="$(redis-cli XLEN signals_stream 2>/dev/null || echo 0)"
  sleep 1
  x2="$(redis-cli XLEN signals_stream 2>/dev/null || echo 0)"
  # XLEN trim ile azalabilir, bu yüzden sadece bilgi amaçlı
  if [[ "$x2" -gt "$x1" ]]; then :; fi
fi

if [[ "$ok" -eq 1 ]]; then
  FAIL_COUNT=0
  LAST_OK="$now"
  echo "[HEALTH][OK] pids ok"
  save_state
  exit 0
fi

# ---- unhealthy -> fail counter ----
FAIL_COUNT=$((FAIL_COUNT + 1))
echo "[HEALTH][WARN] unhealthy (fail_count=${FAIL_COUNT}/${FAIL_THRESH}) reason=${reasons[*]:-unknown}"

# ---- cooldown ----
if (( LAST_RESTART > 0 )) && (( now - LAST_RESTART < COOLDOWN_SEC )); then
  echo "[HEALTH][WARN] cooldown active, skipping restart (age=$((now - LAST_RESTART))s < ${COOLDOWN_SEC}s)"
  save_state
  exit 0
fi

# ---- threshold ----
if (( FAIL_COUNT < FAIL_THRESH )); then
  save_state
  exit 0
fi

echo "[HEALTH][FAIL] threshold reached -> restarting binance1-orch.service (reason=${reasons[*]:-unknown})"
logger -t binance1-orch "HEALTH triggering restart (reason=threshold_reached; details=${reasons[*]:-unknown})"

FAIL_COUNT=0
LAST_RESTART="$now"
save_state

systemctl --user restart binance1-orch.service
exit 0
