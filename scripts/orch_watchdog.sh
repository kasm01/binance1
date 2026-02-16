#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Optional: load env safely
if [[ -f ".env" ]]; then
  ./scripts/load_env.sh .env >/dev/null 2>&1 || true
fi

# -----------------------------
# Single-instance lock
# -----------------------------
exec 9>/tmp/binance1_orch_watchdog.lock
flock -n 9 || exit 0

# -----------------------------
# Redis wrapper (env aware)
# -----------------------------
rc() {
  redis-cli \
    -h "${REDIS_HOST:-127.0.0.1}" \
    -p "${REDIS_PORT:-6379}" \
    -n "${REDIS_DB:-0}" \
    ${REDIS_PASSWORD:+-a "$REDIS_PASSWORD"} \
    "$@"
}

# -----------------------------
# Tuning
# -----------------------------
SLEEP_SEC="${WATCHDOG_SLEEP_SEC:-2}"
MAX_IDLE_SEC="${WATCHDOG_MAX_IDLE_SEC:-45}"
FAIL_THRESHOLD="${WATCHDOG_FAIL_THRESHOLD:-2}"
COOLDOWN_SEC="${WATCHDOG_RESTART_COOLDOWN_SEC:-300}"
GRACE_SEC="${WATCHDOG_GRACE_SEC:-90}"
WARN_COOLDOWN_SEC="${WATCHDOG_WARN_COOLDOWN_SEC:-120}"
OK_COOLDOWN_SEC="${WATCHDOG_OK_COOLDOWN_SEC:-600}"

# Telegram WARN gating:
# - default: only send WARN when FAIL_COUNT reaches threshold-1 (i.e., about to restart)
# - set WATCHDOG_WARN_AT_FAIL_COUNT=1 to warn earlier
WARN_AT_FAIL_COUNT="${WATCHDOG_WARN_AT_FAIL_COUNT:-$((FAIL_THRESHOLD-1))}"
if [[ "$WARN_AT_FAIL_COUNT" -lt 1 ]]; then WARN_AT_FAIL_COUNT=1; fi

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/binance1"
STATE_FILE="$STATE_DIR/orch_watchdog.state"
mkdir -p "$STATE_DIR"

need() {
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "[WD][FAIL] missing: $cmd"; exit 2; }
  done
}
need redis-cli curl pgrep awk date flock logger systemctl ps grep head tr sed

now_ts() { date +%s; }
now_ms() { date +%s%3N; }

# -----------------------------
# Telegram helpers
# -----------------------------
tg_send(){
  local text="$1"
  local tok="${TELEGRAM_BOT_TOKEN:-}"
  local chat="${TELEGRAM_CHAT_ID:-}"
  local enabled="${TELEGRAM_ALERTS:-0}"
  [[ "${enabled}" == "1" ]] || return 0
  [[ -n "${tok}" && -n "${chat}" ]] || return 0
  text="${text//$'\n'/%0A}"
  curl -sS --max-time 5 \
    -X POST "https://api.telegram.org/bot${tok}/sendMessage" \
    -d "chat_id=${chat}" \
    -d "text=${text}" \
    >/dev/null 2>&1 || true
}

tg_warn(){
  local text="$1"
  local ts
  ts="$(now_ts)"
  if [[ "$((ts - LAST_WARN_TS))" -lt "${WARN_COOLDOWN_SEC}" ]]; then
    return 0
  fi
  LAST_WARN_TS="$ts"
  write_state
  tg_send "$text"
}

tg_ok(){
  local text="$1"
  local ts
  ts="$(now_ts)"
  if [[ "$((ts - LAST_RECOVERY_TS))" -lt "${OK_COOLDOWN_SEC}" ]]; then
    return 0
  fi
  LAST_RECOVERY_TS="$ts"
  write_state
  tg_send "$text"
}

# -----------------------------
# State
# -----------------------------
read_state() {
  FAIL_COUNT=0
  LAST_RESTART=0
  LAST_OK=0
  LAST_WARN_TS=0
  LAST_RECOVERY_TS=0
  [[ -f "${STATE_FILE:-}" ]] && source "$STATE_FILE" || true
}

write_state() {
  cat >"${STATE_FILE}.tmp" <<EOF
FAIL_COUNT=$FAIL_COUNT
LAST_RESTART=$LAST_RESTART
LAST_OK=$LAST_OK
LAST_WARN_TS=$LAST_WARN_TS
LAST_RECOVERY_TS=$LAST_RECOVERY_TS
EOF
  mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
}

bump_fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); write_state; }
reset_fail() { FAIL_COUNT=0; LAST_OK="$(now_ts)"; write_state; }

# -----------------------------
# Skip checks if orch service is not active (prevents false alarms during manual stop)
# -----------------------------
if systemctl --user is-active --quiet binance1-orch.service 2>/dev/null; then
  : # active -> continue
else
  exit 0
fi

# -----------------------------
# Startup grace: if pidfiles are very new, skip this run
# -----------------------------
STARTUP_GRACE_SEC="${WATCHDOG_STARTUP_GRACE_SEC:-60}"
if ls run/*.pid >/dev/null 2>&1; then
  newest_pid_mtime="$(stat -c %Y run/*.pid 2>/dev/null | sort -nr | head -n 1 || echo 0)"
  ts_now="$(date +%s)"
  if [[ "$newest_pid_mtime" -gt 0 ]] && [[ "$((ts_now - newest_pid_mtime))" -lt "$STARTUP_GRACE_SEC" ]]; then
    exit 0
  fi
fi

# -----------------------------
# DRY_RUN logic:
# - In DRY_RUN mode exec_events_stream may be sparse; don't require it to advance
# -----------------------------
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CHECK_EXEC_STREAM=0
else
  CHECK_EXEC_STREAM=1
fi

# -----------------------------
# Stream helpers
# -----------------------------
last_id() {
  rc --raw XREVRANGE "$1" + - COUNT 1 2>/dev/null | head -n 1 | tr -d '\r' || true
}

id_ms() { awk -F'-' '{print $1}' <<<"${1:-}" 2>/dev/null || true; }

stream_age_sec(){
  local id="$1"
  local nm ms
  nm="$(now_ms)"
  ms="$(id_ms "$id")"
  [[ -n "${ms:-}" ]] || { echo "na"; return; }
  echo $(( (nm - ms) / 1000 ))
}

stream_snapshot(){
  local sig exe age_sig age_exe
  sig="$(last_id signals_stream)"
  if [[ "$CHECK_EXEC_STREAM" == "1" ]]; then
    exe="$(last_id exec_events_stream)"
  else
    exe="na"
  fi
  age_sig="$(stream_age_sec "$sig")"
  age_exe="$(stream_age_sec "$exe")"
  echo "signals=${sig:-na} age_sig=${age_sig}s | exec=${exe:-na} age_exe=${age_exe}s"
}

emit_health_snapshot() {
  echo "missing=(${missing_proc[*]:-none}) | $(stream_snapshot)"
}

# OK helper: if we were failing before and now recovered, send a recovery message (rate-limited)
mark_ok_and_reset(){
  local prev="$FAIL_COUNT"
  if [[ "$prev" -gt 0 ]]; then
    tg_ok "✅ WATCHDOG OK recovered
$(stream_snapshot)
prev_fail=${prev}/${FAIL_THRESHOLD}"
  fi
  reset_fail
}

# -----------------------------
# Process health
# -----------------------------
expected_cmd() {
  case "$1" in
    scanner_*) echo "orchestration/scanners/worker_stub.py" ;;
    aggregator) echo "orchestration/aggregator/run_aggregator.py" ;;
    top_selector) echo "orchestration/selector/top_selector.py" ;;
    master_executor) echo "orchestration/executor/master_executor.py" ;;
    intent_bridge) echo "orchestration/executor/intent_bridge.py" ;;
    *) echo "" ;;
  esac
}

check_proc_with_pidfile() {
  local name="$1"
  local expect pidfile pid cmd
  expect="$(expected_cmd "$name")"
  pidfile="run/${name}.pid"

  [[ -n "${expect:-}" ]] || return 1
  [[ -f "$pidfile" ]] || return 1

  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1

  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ -n "${cmd:-}" ]] || return 1

  echo "$cmd" | grep -Fq "$expect" || return 1
  return 0
}

# -----------------------------
# Restart logic
# -----------------------------
do_restart_if_needed() {
  local reason="$1"
  local ts age snap
  ts="$(now_ts)"
  age=$((ts - LAST_RESTART))

  if [[ "$age" -lt "$COOLDOWN_SEC" ]]; then
    echo "[WD][WARN] cooldown active (${age}s < ${COOLDOWN_SEC}s)"
    return 0
  fi

  if [[ "$FAIL_COUNT" -lt "$FAIL_THRESHOLD" ]]; then
    echo "[WD][WARN] $reason fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"
    return 0
  fi

  snap="$(emit_health_snapshot)"
  tg_send "⚠️ WATCHDOG restart: ${reason}
${snap}
fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"

  LAST_RESTART="$ts"
  FAIL_COUNT=0
  write_state

  systemctl --user restart binance1-orch.service
  exit 0
}

# -----------------------------
# Start
# -----------------------------
read_state

# Grace period after restart
ts_now="$(now_ts)"
if (( LAST_RESTART > 0 )) && (( ts_now - LAST_RESTART < GRACE_SEC )); then
  reset_fail
  exit 0
fi

# -----------------------------
# Process check
# -----------------------------
must_names=(
  scanner_w1 scanner_w2 scanner_w3 scanner_w4
  scanner_w5 scanner_w6 scanner_w7 scanner_w8
  aggregator top_selector master_executor intent_bridge
)

missing_proc=()
for name in "${must_names[@]}"; do
  if ! check_proc_with_pidfile "$name"; then
    missing_proc+=("${name}")
  fi
done

if [[ "${#missing_proc[@]}" -gt 0 ]]; then
  bump_fail

  # Telegram WARN only when we are close to restart threshold (or configured otherwise)
  if [[ "$FAIL_COUNT" -ge "$WARN_AT_FAIL_COUNT" ]]; then
    tg_warn "⚠️ WATCHDOG WARN missing: ${missing_proc[*]}
$(stream_snapshot)
fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"
  fi

  do_restart_if_needed "missing process(es)"
  exit 0
fi

# -----------------------------
# Stream activity
# -----------------------------
sig1="$(last_id signals_stream)"
if [[ "$CHECK_EXEC_STREAM" == "1" ]]; then
  exe1="$(last_id exec_events_stream)"
else
  exe1="na"
fi

sleep "$SLEEP_SEC"

sig2="$(last_id signals_stream)"
if [[ "$CHECK_EXEC_STREAM" == "1" ]]; then
  exe2="$(last_id exec_events_stream)"
else
  exe2="na"
fi

# Activity rule:
# - Always accept signals advancing.
# - Accept exec advancing only if CHECK_EXEC_STREAM=1.
if [[ "$sig2" != "$sig1" ]]; then
  mark_ok_and_reset
  exit 0
fi
if [[ "$CHECK_EXEC_STREAM" == "1" && "$exe2" != "$exe1" ]]; then
  mark_ok_and_reset
  exit 0
fi

age_sig="$(stream_age_sec "$sig2")"
age_exe="$(stream_age_sec "$exe2")"

# Age rule:
# - signals fresh => OK
# - exec fresh => OK only if CHECK_EXEC_STREAM=1
if [[ "$age_sig" != "na" && "$age_sig" -le "$MAX_IDLE_SEC" ]]; then
  mark_ok_and_reset
  exit 0
fi
if [[ "$CHECK_EXEC_STREAM" == "1" && "$age_exe" != "na" && "$age_exe" -le "$MAX_IDLE_SEC" ]]; then
  mark_ok_and_reset
  exit 0
fi

bump_fail

# Telegram WARN only when we are close to restart threshold (or configured otherwise)
if [[ "$FAIL_COUNT" -ge "$WARN_AT_FAIL_COUNT" ]]; then
  tg_warn "⚠️ WATCHDOG WARN no stream activity
$(stream_snapshot)
fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"
fi

do_restart_if_needed "no stream activity"
exit 0

