#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# -----------------------------
# Load .env into THIS shell (and export to children) - deterministic
# - This guarantees watchdog sees same env as orch_start.
# - We do NOT rely solely on load_env.sh (which may be "non-override" and miss vars in systemd context).
# -----------------------------
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# Optional: keep legacy loader (no-op is fine)
if [[ -x "./scripts/load_env.sh" ]] && [[ -f ".env" ]]; then
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

# Stream watchdog / auto recovery
STREAM_WATCHDOG_ENABLE="${STREAM_WATCHDOG_ENABLE:-1}"
STREAM_WATCHDOG_IDLE_SEC="${STREAM_WATCHDOG_IDLE_SEC:-90}"
STREAM_WATCHDOG_MIN_XLEN="${STREAM_WATCHDOG_MIN_XLEN:-1}"
STREAM_WATCHDOG_WARN_ONLY="${STREAM_WATCHDOG_WARN_ONLY:-0}"

STREAM_WATCHDOG_CHECK_AGG="${STREAM_WATCHDOG_CHECK_AGG:-1}"
STREAM_WATCHDOG_CHECK_SELECTOR="${STREAM_WATCHDOG_CHECK_SELECTOR:-1}"
STREAM_WATCHDOG_CHECK_MASTER="${STREAM_WATCHDOG_CHECK_MASTER:-1}"
STREAM_WATCHDOG_CHECK_BRIDGE="${STREAM_WATCHDOG_CHECK_BRIDGE:-1}"

STREAM_RECOVERY_TARGETED_ENABLE="${STREAM_RECOVERY_TARGETED_ENABLE:-1}"
STREAM_RECOVERY_COOLDOWN_SEC="${STREAM_RECOVERY_COOLDOWN_SEC:-180}"
STREAM_RECOVERY_TOUCH_STATE="${STREAM_RECOVERY_TOUCH_STATE:-1}"

STREAM_GROUP_HEAL_ENABLE="${STREAM_GROUP_HEAL_ENABLE:-1}"
STREAM_GROUP_HEAL_PENDING_MIN="${STREAM_GROUP_HEAL_PENDING_MIN:-20}"
STREAM_GROUP_HEAL_IDLE_SEC="${STREAM_GROUP_HEAL_IDLE_SEC:-120}"
STREAM_GROUP_HEAL_COOLDOWN_SEC="${STREAM_GROUP_HEAL_COOLDOWN_SEC:-300}"
STREAM_GROUP_HEAL_DESTROY_RECREATE="${STREAM_GROUP_HEAL_DESTROY_RECREATE:-0}"

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/binance1"
STATE_FILE="$STATE_DIR/orch_watchdog.state"
STREAM_RECOVERY_STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/binance1_stream_recovery.state"
mkdir -p "$STATE_DIR"

need() {
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "[WD][FAIL] missing: $cmd"; exit 2; }
  done
}
need redis-cli curl pgrep awk date flock logger systemctl ps grep head tr sed stat sort ls

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

read_stream_recovery_state() {
  LAST_STREAM_RECOVERY_TS=0
  LAST_STREAM_RECOVERY_LABEL=""
  [[ -f "$STREAM_RECOVERY_STATE_FILE" ]] && source "$STREAM_RECOVERY_STATE_FILE" 2>/dev/null || true
}

write_stream_recovery_state() {
  local ts="$1"
  local label="$2"
  cat > "${STREAM_RECOVERY_STATE_FILE}.tmp" <<EOF
LAST_STREAM_RECOVERY_TS=$ts
LAST_STREAM_RECOVERY_LABEL="$label"
EOF
  mv -f "${STREAM_RECOVERY_STATE_FILE}.tmp" "$STREAM_RECOVERY_STATE_FILE"
}

stream_recovery_allowed() {
  read_stream_recovery_state
  local ts_now delta
  ts_now="$(now_ts)"
  delta=$(( ts_now - ${LAST_STREAM_RECOVERY_TS:-0} ))
  (( delta >= STREAM_RECOVERY_COOLDOWN_SEC ))
}
STREAM_GROUP_HEAL_STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/binance1_stream_group_heal.state"

read_group_heal_state() {
  LAST_GROUP_HEAL_TS=0
  LAST_GROUP_HEAL_TARGET=""
  [[ -f "$STREAM_GROUP_HEAL_STATE_FILE" ]] && source "$STREAM_GROUP_HEAL_STATE_FILE" 2>/dev/null || true
}

write_group_heal_state() {
  local ts="$1"
  local target="$2"
  cat > "${STREAM_GROUP_HEAL_STATE_FILE}.tmp" <<EOF
LAST_GROUP_HEAL_TS=$ts
LAST_GROUP_HEAL_TARGET="$target"
EOF
  mv -f "${STREAM_GROUP_HEAL_STATE_FILE}.tmp" "$STREAM_GROUP_HEAL_STATE_FILE"
}

group_heal_allowed() {
  read_group_heal_state
  local ts_now delta
  ts_now="$(now_ts)"
  delta=$(( ts_now - ${LAST_GROUP_HEAL_TS:-0} ))
  (( delta >= STREAM_GROUP_HEAL_COOLDOWN_SEC ))
}

stream_group_pending_count() {
  local stream="$1"
  local group="$2"
  rc XPENDING "$stream" "$group" 2>/dev/null | head -n 1 | tr -d '\r' | awk '{print $1+0}'
}

stream_group_last_delivered_id() {
  local stream="$1"
  local group="$2"
  rc XINFO GROUPS "$stream" 2>/dev/null \
    | awk -v g="$group" '
      $1=="name" {name=$2; gsub(/"/,"",name)}
      $1=="last-delivered-id" && name==g {print $2; gsub(/"/,""); exit}
    ' \
    | tr -d '"' | tr -d '\r'
}

stream_group_last_delivered_age_sec() {
  local stream="$1"
  local group="$2"
  local id
  id="$(stream_group_last_delivered_id "$stream" "$group")"
  stream_age_sec "$id"
}

heal_stream_group_target() {
  local label="$1"
  local stream="$2"
  local group="$3"

  echo "[WATCHDOG] group self-heal -> label=${label} stream=${stream} group=${group}" >&2

  case "$label" in
    selector)
      pkill -f "orchestration/selector/top_selector.py" 2>/dev/null || true
      ;;
    master)
      pkill -f "orchestration/executor/master_executor.py" 2>/dev/null || true
      ;;
    bridge)
      pkill -f "orchestration/executor/intent_bridge.py" 2>/dev/null || true
      ;;
    agg)
      pkill -f "orchestration/aggregator/run_aggregator.py" 2>/dev/null || true
      ;;
    *)
      return 1
      ;;
  esac

  sleep 2

  if [[ "$STREAM_GROUP_HEAL_DESTROY_RECREATE" == "1" ]]; then
    rc XGROUP DESTROY "$stream" "$group" >/dev/null 2>&1 || true
    rc XGROUP CREATE "$stream" "$group" '$' MKSTREAM >/dev/null 2>&1 || true
  fi

  return 0
}
# -----------------------------
# Skip checks if orch service is not active (prevents false alarms during manual stop)
# -----------------------------
if systemctl --user is-active --quiet binance1-orch.service 2>/dev/null; then
  :
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
# DRY_RUN logic
# -----------------------------
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CHECK_EXEC_STREAM="${WATCHDOG_CHECK_EXEC_IN_DRYRUN:-0}"
else
  CHECK_EXEC_STREAM=1
fi

# -----------------------------
# Stream helpers (robust)
# -----------------------------
last_generated_id() {
  local stream="$1"
  local id
  id="$(rc XINFO STREAM "$stream" 2>/dev/null \
        | awk '$1=="last-generated-id"{print $2; exit}' \
        | tr -d '"' | tr -d '\r' || true)"
  if [[ -n "${id:-}" ]]; then
    echo "$id"
    return 0
  fi
  rc --raw XREVRANGE "$stream" + - COUNT 1 2>/dev/null | head -n 1 | tr -d '\r' || true
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

stream_xlen() {
  local stream="$1"
  rc XLEN "$stream" 2>/dev/null | tr -d '\r' | awk '{print $1+0}'
}

stream_snapshot(){
  local sig exe age_sig age_exe
  sig="$(last_generated_id signals_stream)"
  if [[ "$CHECK_EXEC_STREAM" == "1" ]]; then
    exe="$(last_generated_id exec_events_stream)"
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

mark_ok_and_reset(){
  local prev="$FAIL_COUNT"
  if [[ "$prev" -gt 0 ]]; then
    tg_ok "✅ WATCHDOG OK recovered
$(stream_snapshot)
prev_fail=${prev}/${FAIL_THRESHOLD}"
  fi
  reset_fail
}
watch_stream_pair() {
  local upstream="$1"
  local downstream="$2"
  local label="$3"

  local up_id dn_id up_age dn_age up_xlen dn_xlen
  up_id="$(last_generated_id "$upstream")"
  dn_id="$(last_generated_id "$downstream")"

  up_age="$(stream_age_sec "$up_id")"
  dn_age="$(stream_age_sec "$dn_id")"

  up_xlen="$(stream_xlen "$upstream")"
  dn_xlen="$(stream_xlen "$downstream")"

  [[ -n "${up_xlen:-}" ]] || up_xlen=0
  [[ -n "${dn_xlen:-}" ]] || dn_xlen=0

  if [[ "$up_xlen" -lt "$STREAM_WATCHDOG_MIN_XLEN" ]]; then
    return 0
  fi

  if [[ "$up_age" != "na" && "$up_age" -le "$STREAM_WATCHDOG_IDLE_SEC" ]]; then
    if [[ "$dn_age" == "na" || "$dn_age" -gt "$STREAM_WATCHDOG_IDLE_SEC" ]]; then
      echo "${label}|up=${upstream}|dn=${downstream}|up_age=${up_age}|dn_age=${dn_age}|up_xlen=${up_xlen}|dn_xlen=${dn_xlen}"
      return 1
    fi
  fi

  return 0
}

extract_stream_failure_label() {
  local msg="$1"
  echo "$msg" | awk -F'|' '{print $1}'
}

restart_target_for_label() {
  local label="$1"

  case "$label" in
    agg)
      echo "[WATCHDOG] targeted recovery -> aggregator" >&2
      pkill -f "orchestration/aggregator/run_aggregator.py" 2>/dev/null || true
      ;;
    selector)
      echo "[WATCHDOG] targeted recovery -> top_selector" >&2
      pkill -f "orchestration/selector/top_selector.py" 2>/dev/null || true
      ;;
    master)
      echo "[WATCHDOG] targeted recovery -> master_executor" >&2
      pkill -f "orchestration/executor/master_executor.py" 2>/dev/null || true
      ;;
    bridge)
      echo "[WATCHDOG] targeted recovery -> intent_bridge" >&2
      pkill -f "orchestration/executor/intent_bridge.py" 2>/dev/null || true
      ;;
    *)
      return 1
      ;;
  esac

  sleep 2
  return 0
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
must_names=()

for n in 1 2 3 4 5 6 7 8; do
  var="W${n}_SYMBOLS"
  syms="$(eval echo \${$var})"
  syms="$(echo "$syms" | xargs 2>/dev/null || true)"

  if [[ -n "${syms:-}" ]]; then
    must_names+=("scanner_w${n}")
  fi
done

must_names+=(
  aggregator
  top_selector
  master_executor
  intent_bridge
)
missing_proc=()
for name in "${must_names[@]}"; do
  if ! check_proc_with_pidfile "$name"; then
    missing_proc+=("${name}")
  fi
done

if [[ "${#missing_proc[@]}" -gt 0 ]]; then
  bump_fail

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
sig1="$(last_generated_id signals_stream)"
if [[ "$CHECK_EXEC_STREAM" == "1" ]]; then
  exe1="$(last_generated_id exec_events_stream)"
else
  exe1="na"
fi

sleep "$SLEEP_SEC"

sig2="$(last_generated_id signals_stream)"
if [[ "$CHECK_EXEC_STREAM" == "1" ]]; then
  exe2="$(last_generated_id exec_events_stream)"
else
  exe2="na"
fi

if [[ "$sig2" != "$sig1" ]]; then
  mark_ok_and_reset
else
  if [[ "$CHECK_EXEC_STREAM" == "1" && "$exe2" != "$exe1" ]]; then
    mark_ok_and_reset
  else
    age_sig="$(stream_age_sec "$sig2")"
    age_exe="$(stream_age_sec "$exe2")"

    if [[ "$age_sig" != "na" && "$age_sig" -le "$MAX_IDLE_SEC" ]]; then
      mark_ok_and_reset
    elif [[ "$CHECK_EXEC_STREAM" == "1" && "$age_exe" != "na" && "$age_exe" -le "$MAX_IDLE_SEC" ]]; then
      mark_ok_and_reset
    else
      bump_fail

      if [[ "$FAIL_COUNT" -ge "$WARN_AT_FAIL_COUNT" ]]; then
        tg_warn "⚠️ WATCHDOG WARN no stream activity
$(stream_snapshot)
fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"
      fi

      do_restart_if_needed "no stream activity"
      exit 0
    fi
  fi
fi

# -----------------------------
# Stream chain watchdog
# -----------------------------
if [[ "$STREAM_WATCHDOG_ENABLE" == "1" ]]; then
  stream_chain_failures=()

  if [[ "$STREAM_WATCHDOG_CHECK_AGG" == "1" ]]; then
    if ! msg="$(watch_stream_pair "signals_stream" "candidates_stream" "agg")"; then
      stream_chain_failures+=("$msg")
    fi
  fi

  if [[ "$STREAM_WATCHDOG_CHECK_SELECTOR" == "1" ]]; then
    if ! msg="$(watch_stream_pair "candidates_stream" "top5_stream" "selector")"; then
      stream_chain_failures+=("$msg")
    fi
  fi

  if [[ "$STREAM_WATCHDOG_CHECK_MASTER" == "1" ]]; then
    if ! msg="$(watch_stream_pair "top5_stream" "trade_intents_stream" "master")"; then
      stream_chain_failures+=("$msg")
    fi
  fi

  if [[ "$STREAM_WATCHDOG_CHECK_BRIDGE" == "1" ]]; then
    if ! msg="$(watch_stream_pair "trade_intents_stream" "exec_events_stream" "bridge")"; then
      stream_chain_failures+=("$msg")
    fi
  fi

  if [[ "${#stream_chain_failures[@]}" -gt 0 ]]; then
    bump_fail

    if [[ "$FAIL_COUNT" -ge "$WARN_AT_FAIL_COUNT" ]]; then
      tg_warn "⚠️ STREAM WATCHDOG WARN
$(printf '%s\n' "${stream_chain_failures[@]}")
$(stream_snapshot)
fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"
    fi

    if [[ "$STREAM_WATCHDOG_WARN_ONLY" != "1" ]]; then
      if [[ "$STREAM_GROUP_HEAL_ENABLE" == "1" ]]; then
        first_label="$(extract_stream_failure_label "${stream_chain_failures[0]}")"

        heal_stream=""
        heal_group=""

        case "$first_label" in
          agg)
            heal_stream="candidates_stream"
            heal_group="agg_group"
            ;;
          selector)
            heal_stream="candidates_stream"
            heal_group="topsel_g"
            ;;
          master)
            heal_stream="top5_stream"
            heal_group="master_exec_g"
            ;;
          bridge)
            heal_stream="trade_intents_stream"
            heal_group="bridge_g"
            ;;
        esac

        if [[ -n "$heal_stream" && -n "$heal_group" ]]; then
          pending_cnt="$(stream_group_pending_count "$heal_stream" "$heal_group")"
          delivered_age="$(stream_group_last_delivered_age_sec "$heal_stream" "$heal_group")"

          [[ -n "${pending_cnt:-}" ]] || pending_cnt=0
          [[ -n "${delivered_age:-}" ]] || delivered_age="na"

          if group_heal_allowed; then
            if [[ "$delivered_age" != "na" && "$delivered_age" -ge "$STREAM_GROUP_HEAL_IDLE_SEC" ]] || \
               [[ "$pending_cnt" -ge "$STREAM_GROUP_HEAL_PENDING_MIN" ]]; then

              if heal_stream_group_target "$first_label" "$heal_stream" "$heal_group"; then
                write_group_heal_state "$(now_ts)" "${first_label}:${heal_stream}:${heal_group}"

                tg_warn "♻️ STREAM GROUP SELF-HEAL
label=${first_label}
stream=${heal_stream}
group=${heal_group}
pending=${pending_cnt}
last_delivered_age=${delivered_age}s
$(printf '%s\n' "${stream_chain_failures[@]}")
$(stream_snapshot)"

                exit 0
              fi
            fi
          fi
        fi
      fi

      if [[ "$STREAM_RECOVERY_TARGETED_ENABLE" == "1" ]]; then
        first_label="$(extract_stream_failure_label "${stream_chain_failures[0]}")"

        if stream_recovery_allowed; then
          if restart_target_for_label "$first_label"; then
            if [[ "$STREAM_RECOVERY_TOUCH_STATE" == "1" ]]; then
              write_stream_recovery_state "$(now_ts)" "$first_label"
            fi

            tg_warn "🛠️ STREAM TARGETED RECOVERY
label=${first_label}
$(printf '%s\n' "${stream_chain_failures[@]}")
$(stream_snapshot)"

            exit 0
          fi
        fi
      fi

      do_restart_if_needed "stream chain stalled"
      exit 0
    fi
  fi
fi

exit 0
