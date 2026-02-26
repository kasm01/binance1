#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/binance1"
STATE_FILE="$STATE_DIR/orch_health.state"
mkdir -p "$STATE_DIR"

# ---- Tuning (override) ----
GRACE_SEC="${HEALTH_GRACE_SEC:-90}"
FAIL_THRESH="${HEALTH_FAIL_THRESH:-3}"
COOLDOWN_SEC="${HEALTH_COOLDOWN_SEC:-300}"

REDIS_DB="${REDIS_DB:-0}"

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

expected_cmd() {
  case "${1:-}" in
    aggregator)      echo "orchestration/aggregator/run_aggregator.py" ;;
    top_selector)    echo "orchestration/selector/top_selector.py" ;;
    master_executor) echo "orchestration/executor/master_executor.py" ;;
    intent_bridge)   echo "orchestration/executor/intent_bridge.py" ;;
    *)               echo "" ;;
  esac
}

reasons=()

check_proc_with_pidfile() {
  local n="$1"
  local pidfile="run/${n}.pid"
  local expect pid cmd newpid
  expect="$(expected_cmd "$n")"

  if [[ ! -f "$pidfile" ]]; then
    if [[ -n "${expect:-}" ]]; then
      newpid="$(pgrep -f "$expect" | head -n1 || true)"
      if [[ -n "${newpid:-}" ]]; then
        echo "$newpid" > "$pidfile"
        echo "[HEALTH][WARN] ${n} pidfile_missing -> healed (new=${newpid})"
        return 0
      fi
    fi
    reasons+=("${n}:pidfile_missing")
    return 1
  fi

  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "${pid:-}" ]]; then
    if [[ -n "${expect:-}" ]]; then
      newpid="$(pgrep -f "$expect" | head -n1 || true)"
      if [[ -n "${newpid:-}" ]]; then
        echo "$newpid" > "$pidfile"
        echo "[HEALTH][WARN] ${n} pid_empty -> healed (new=${newpid})"
        return 0
      fi
    fi
    reasons+=("${n}:pid_empty")
    return 1
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    if [[ -n "${expect:-}" ]]; then
      newpid="$(pgrep -f "$expect" | head -n1 || true)"
      if [[ -n "${newpid:-}" ]]; then
        echo "$newpid" > "$pidfile"
        echo "[HEALTH][WARN] ${n} pid_dead(${pid}) -> healed (new=${newpid})"
        return 0
      fi
    fi
    reasons+=("${n}:pid_dead(${pid})")
    return 1
  fi

  if [[ -n "${expect:-}" ]]; then
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ -n "${cmd:-}" ]] && echo "$cmd" | grep -Fq "$expect"; then
      return 0
    fi
    newpid="$(pgrep -f "$expect" | head -n1 || true)"
    if [[ -n "${newpid:-}" ]]; then
      echo "$newpid" > "$pidfile"
      echo "[HEALTH][WARN] ${n} pid_cmd_mismatch(pid=${pid}) -> healed (new=${newpid})"
      return 0
    fi
    reasons+=("${n}:pid_cmd_mismatch(${pid})")
    return 1
  fi

  return 0
}

need_names=(aggregator top_selector master_executor intent_bridge)

ok=1
for n in "${need_names[@]}"; do
  if ! check_proc_with_pidfile "$n"; then
    ok=0
  fi
done

# soft redis check (never fails health alone)
if need redis-cli; then
  redis-cli -n "$REDIS_DB" PING >/dev/null 2>&1 || true
fi

if [[ "$ok" -eq 1 ]]; then
  FAIL_COUNT=0
  LAST_OK="$now"
  echo "[HEALTH][OK] pids ok"
  save_state
  exit 0
fi

FAIL_COUNT=$((FAIL_COUNT + 1))
echo "[HEALTH][WARN] unhealthy (fail_count=${FAIL_COUNT}/${FAIL_THRESH}) reason=${reasons[*]:-unknown}"

if (( LAST_RESTART > 0 )) && (( now - LAST_RESTART < COOLDOWN_SEC )); then
  echo "[HEALTH][WARN] cooldown active, skipping restart (age=$((now - LAST_RESTART))s < ${COOLDOWN_SEC}s)"
  save_state
  exit 0
fi

if (( FAIL_COUNT < FAIL_THRESH )); then
  save_state
  exit 0
fi

echo "[HEALTH][FAIL] threshold reached -> restarting binance1-orch.service (reason=${reasons[*]:-unknown})"
logger -t binance1-orch "HEALTH triggering restart (details=${reasons[*]:-unknown})" || true

FAIL_COUNT=0
LAST_RESTART="$now"
save_state

systemctl --user restart binance1-orch.service
exit 0
