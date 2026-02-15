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
# Tuning
# -----------------------------
SLEEP_SEC="${WATCHDOG_SLEEP_SEC:-2}"
MAX_IDLE_SEC="${WATCHDOG_MAX_IDLE_SEC:-45}"          # growth yoksa bile son event <=45s ise OK (env ile override)
FAIL_THRESHOLD="${WATCHDOG_FAIL_THRESHOLD:-2}"       # restart için ardışık fail
COOLDOWN_SEC="${WATCHDOG_RESTART_COOLDOWN_SEC:-300}" # restartlar arası min süre
GRACE_SEC="${WATCHDOG_GRACE_SEC:-90}"                # restart/start sonrası grace (false alarm keser)
WARN_COOLDOWN_SEC="${WATCHDOG_WARN_COOLDOWN_SEC:-300}"   # WARN telegram spam engeli
OK_COOLDOWN_SEC="${WATCHDOG_OK_COOLDOWN_SEC:-600}"       # OK/recovery spam engeli

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/binance1"
STATE_FILE="$STATE_DIR/orch_watchdog.state"
mkdir -p "$STATE_DIR"

need() { command -v "$1" >/dev/null 2>&1 || { echo "[WD][FAIL] missing: $1"; exit 2; }; }
need redis-cli
need curl
need pgrep
need awk
need date
need flock
need logger
need systemctl
need ps
need grep
need head
need tr
need sed



tg_send(){
  # Send a short alert to Telegram if enabled + token/chat exist
  local text="$1"
  local tok="${TELEGRAM_BOT_TOKEN:-}"
  local chat="${TELEGRAM_CHAT_ID:-}"
  local enabled="${TELEGRAM_ALERTS:-0}"
  [[ "${enabled}" == "1" ]] || return 0
  [[ -n "${tok}" && -n "${chat}" ]] || return 0
  # minimal newline escape
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
  # spam engeli
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
  # spam engeli
  if [[ "$((ts - LAST_RECOVERY_TS))" -lt "${OK_COOLDOWN_SEC}" ]]; then
    return 0
  fi
  LAST_RECOVERY_TS="$ts"
  write_state
  tg_send "$text"
}

# WATCHDOG_TEST_ALERT=1 -> always send a test message (manual verification)
if [[ "${WATCHDOG_TEST_ALERT:-0}" == "1" ]]; then
  tg_send "🧪 WATCHDOG test alert: orch_watchdog.sh is running"
fi

now_ts() { date +%s; }
now_ms() { date +%s%3N; }

read_state() {
  FAIL_COUNT=0
  LAST_RESTART=0
  LAST_OK=0
  LAST_WARN_TS=0
  LAST_RECOVERY_TS=0

  if [[ -f "${STATE_FILE:-}" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE" || true
  fi

  FAIL_COUNT="${FAIL_COUNT:-0}"
  LAST_RESTART="${LAST_RESTART:-0}"
  LAST_OK="${LAST_OK:-0}"
  LAST_WARN_TS="${LAST_WARN_TS:-0}"
  LAST_RECOVERY_TS="${LAST_RECOVERY_TS:-0}"
}

write_state() {
  local tmp="${STATE_FILE}.tmp"
  cat >"$tmp" <<EOF
FAIL_COUNT=$FAIL_COUNT
LAST_RESTART=$LAST_RESTART
LAST_OK=$LAST_OK
LAST_WARN_TS=$LAST_WARN_TS
LAST_RECOVERY_TS=$LAST_RECOVERY_TS
EOF
  mv -f "$tmp" "$STATE_FILE"
}

bump_fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  write_state
}

reset_fail() {
  FAIL_COUNT=0
  LAST_OK="$(now_ts)"
  write_state
}

do_restart_if_needed() {
  local reason="$1"
  local ts age
  ts="$(now_ts)"
  age=$((ts - LAST_RESTART))

  # cooldown: restart storm engeli
  if [[ "$age" -lt "$COOLDOWN_SEC" ]]; then
    echo "[WD][WARN] $reason but in cooldown (${age}s < ${COOLDOWN_SEC}s). fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"
    return 0
  fi

  # eşik dolmadan restart yok
  if [[ "$FAIL_COUNT" -lt "$FAIL_THRESHOLD" ]]; then
    echo "[WD][WARN] $reason. fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD} -> not restarting yet"
    return 0
  fi

  echo "[WD][FAIL] $reason -> restarting binance1-orch.service"
  logger -t binance1-orch "WATCHDOG triggering restart (reason=${reason})"
  snap="$(stream_snapshot)"
  tg_send "⚠️ WATCHDOG restart: ${reason}\n${snap}\nfail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"

  LAST_RESTART="$ts"
  FAIL_COUNT=0
  write_state

  systemctl --user restart binance1-orch.service
  exit 0
}

# -----------------------------
# Stream helpers
# -----------------------------
last_id() {
  # --raw: id satırını temiz alır (1) "..." gibi formatları azaltır
  # Çıkış boşsa '' döner
  redis-cli --raw XREVRANGE "$1" + - COUNT 1 2>/dev/null | head -n 1 | tr -d '\r' || true
}

id_ms() {
  awk -F'-' '{print $1}' <<<"${1:-}" 2>/dev/null || true
}
age_s_from_id() {
  local mid="${1:-}"
  local ms
  ms="$(id_ms "$mid")"
  [[ -n "${ms:-}" ]] || { echo "na"; return 0; }
  local now
  now="$(now_ms)"
  echo $(( (now - ms) / 1000 ))
}

emit_health_snapshot() {
  # Produces: snapshot text for telegram/log
  local sig exe sig_age exe_age
  sig="$(last_id signals_stream)"
  exe="$(last_id exec_events_stream)"

  sig_age="$(age_s_from_id "$sig")"
  exe_age="$(age_s_from_id "$exe")"

  echo "missing=(${missing_proc[*]:-none}) | signals=${sig:-na} age=${sig_age}s | exec=${exe:-na} age=${exe_age}s"
}

stream_age_sec(){
  local id="$1"
  local nm ms
  nm="$(now_ms)"
  ms="$(id_ms "$id")"
  [[ -n "${ms:-}" ]] || { echo "na"; return 0; }
  echo $(( (nm - ms) / 1000 ))
}

stream_snapshot(){
  # outputs short diagnostic string for telegram
  local sig exe age_sig age_exe
  sig="$(last_id signals_stream)"
  exe="$(last_id exec_events_stream)"
  age_sig="$(stream_age_sec "$sig")"
  age_exe="$(stream_age_sec "$exe")"
  echo "signals=${sig:-na} age_sig=${age_sig}s | exec=${exe:-na} age_exe=${age_exe}s"
}

# -----------------------------
# Process health helpers (pidfile-aware + heal)
# -----------------------------

expected_worker_id() {
  # scanner_w1 -> w1 ... scanner_w8 -> w8
  local name="${1:-}"
  case "$name" in
    scanner_w1) echo "w1" ;;
    scanner_w2) echo "w2" ;;
    scanner_w3) echo "w3" ;;
    scanner_w4) echo "w4" ;;
    scanner_w5) echo "w5" ;;
    scanner_w6) echo "w6" ;;
    scanner_w7) echo "w7" ;;
    scanner_w8) echo "w8" ;;
    *) echo "" ;;
  esac
}

pid_has_worker_id() {
  # Usage: pid_has_worker_id <pid> <wid>
  local pid="$1"
  local wid="$2"
  [[ -n "${pid:-}" && -n "${wid:-}" ]] || return 1
  [[ -r "/proc/${pid}/environ" ]] || return 1
  tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep -qx "WORKER_ID=${wid}"
}

find_pid_by_expect_for_name() {
  # Usage: find_pid_by_expect_for_name <name> <expect>
  # If scanner_*: also requires WORKER_ID match.
  local name="$1"
  local expect="$2"
  local wid pid

  wid="$(expected_worker_id "$name")"

  # candidates: all matching python args
  while read -r pid _rest; do
    [[ -n "${pid:-}" ]] || continue
    if [[ -n "${wid:-}" ]]; then
      if pid_has_worker_id "$pid" "$wid"; then
        echo "$pid"
        return 0
      fi
    else
      echo "$pid"
      return 0
    fi
  done < <(pgrep -af "python.*${expect}" 2>/dev/null || true)

  return 1
}

expected_cmd() {
  local name="${1:-}"
  case "$name" in
    scanner_*)       echo "orchestration/scanners/worker_stub.py" ;;
    aggregator)      echo "orchestration/aggregator/run_aggregator.py" ;;
    top_selector)    echo "orchestration/selector/top_selector.py" ;;
    master_executor) echo "orchestration/executor/master_executor.py" ;;
    intent_bridge)   echo "orchestration/executor/intent_bridge.py" ;;
    *)               echo "" ;;
  esac
}

find_pid_by_expect_unused() {
  local expect="$1"
  pgrep -af "python.*${expect}" 2>/dev/null | awk '{print $1}' | head -n1 || true
}

check_proc_with_pidfile() {
  local name="$1"
  local expect pidfile pid cmd newpid

  expect="$(expected_cmd "$name")"
  pidfile="run/${name}.pid"

  if [[ -z "${expect:-}" ]]; then
    echo "[WD][WARN] ${name} no_expected_cmd"
    return 1
  fi

  # pidfile yoksa -> pgrep ile heal
  if [[ ! -f "$pidfile" ]]; then
    newpid="$(find_pid_by_expect_for_name "$name" "$expect" || true)"
    if [[ -n "${newpid:-}" ]]; then
      echo "$newpid" > "$pidfile"
      echo "[WD][WARN] ${name} pidfile_missing -> healed (new=${newpid})"
      return 0
    fi
    echo "[WD][WARN] ${name} pidfile_missing and cannot heal"
    return 1
  fi

  pid="$(cat "$pidfile" 2>/dev/null || true)"

  # boş pid -> heal
  if [[ -z "${pid:-}" ]]; then
    newpid="$(find_pid_by_expect_for_name "$name" "$expect" || true)"
    if [[ -n "${newpid:-}" ]]; then
      echo "$newpid" > "$pidfile"
      echo "[WD][WARN] ${name} pid_empty -> healed (new=${newpid})"
      return 0
    fi
    echo "[WD][WARN] ${name} pid_empty and cannot heal"
    return 1
  fi

  # pid ölü -> heal
  if ! kill -0 "$pid" 2>/dev/null; then
    newpid="$(find_pid_by_expect_for_name "$name" "$expect" || true)"
    if [[ -n "${newpid:-}" ]]; then
      echo "$newpid" > "$pidfile"
      echo "[WD][WARN] ${name} pid_dead(${pid}) -> healed (new=${newpid})"
      return 0
    fi
    echo "[WD][WARN] ${name} pid_dead(${pid}) and cannot heal"
    return 1
  fi

  # pid canlı -> cmd doğrula
  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ -n "${cmd:-}" ]] && echo "$cmd" | grep -Fq "$expect"; then
    return 0
  fi

  # mismatch -> heal
  newpid="$(find_pid_by_expect_for_name "$name" "$expect" || true)"
  if [[ -n "${newpid:-}" ]]; then
    echo "$newpid" > "$pidfile"
    echo "[WD][WARN] ${name} pid_cmd_mismatch(pid=${pid}) -> healed (new=${newpid})"
    return 0
  fi

  echo "[WD][WARN] ${name} pid_cmd_mismatch(pid=${pid}) and cannot heal"
  return 1
}

# -----------------------------
# Start
# -----------------------------
read_state

# -----------------------------
# TEST: simulate fail alert (NO restart)
# WATCHDOG_TEST_FAIL_ALERT=1 -> sends telegram once and exits
# -----------------------------
if [[ "${WATCHDOG_TEST_FAIL_ALERT:-0}" == "1" ]]; then
  tg_send "🧪 WATCHDOG simulated FAIL alert: would restart service if threshold met"
  echo "[WD][TEST] simulated fail alert sent (no restart)."
  exit 0
fi

# -----------------------------
# Grace (restart sonrası false alarm keser)
# -----------------------------
ts_now="$(now_ts)"
if (( LAST_RESTART > 0 )) && (( ts_now - LAST_RESTART < GRACE_SEC )); then
  age_grace=$((ts_now - LAST_RESTART))
  echo "[WD][OK] in grace period (age=${age_grace}s < ${GRACE_SEC}s)"

  # Restart sonrası ilk grace döneminde 1 kere "OK" bildirimi
  # (LAST_OK, reset_fail içinde güncellendiği için spam olmaz)
  snap="$(emit_health_snapshot)"
  tg_send "✅ WATCHDOG OK (post-restart): ${snap}"

  reset_fail
  exit 0
fi

# -----------------------------
# Process health FIRST (strict)
# -----------------------------
must_names=(
  scanner_w1 scanner_w2 scanner_w3 scanner_w4
  scanner_w5 scanner_w6 scanner_w7 scanner_w8
  aggregator top_selector master_executor intent_bridge
)

missing_proc=()
for name in "${must_names[@]}"; do
  expect="$(expected_cmd "$name")"
  if [[ -z "${expect:-}" ]]; then
    missing_proc+=("${name}:no_expected_cmd")
    continue
  fi

  if ! check_proc_with_pidfile "$name"; then
    missing_proc+=("${name}:missing_or_unhealthy")
  fi
done

if [[ "${#missing_proc[@]}" -gt 0 ]]; then
  bump_fail

  # --- Early WARN (even before restart threshold) ---
  # throttle to avoid spam
  WARN_COOLDOWN_SEC="${WATCHDOG_WARN_COOLDOWN_SEC:-120}"
  warn_key="binance1_wd_warn_missing"
  last_warn_ts="$(redis-cli GET "${warn_key}" 2>/dev/null || echo "")"
  now_warn_ts="$(now_ts)"

  should_warn=1
  if [[ -n "${last_warn_ts:-}" ]]; then
    if (( now_warn_ts - last_warn_ts < WARN_COOLDOWN_SEC )); then
      should_warn=0
    fi
  fi

  if [[ "${should_warn}" -eq 1 ]]; then
    # set/update warn timestamp (best effort)
    redis-cli SET "${warn_key}" "${now_warn_ts}" EX "${WARN_COOLDOWN_SEC}" >/dev/null 2>&1 || true

    snap="$(stream_snapshot)"
    tg_warn "⚠️ WATCHDOG WARN: missing/unhealthy process(es): ${missing_proc[*]}
${snap}
fail_count=${FAIL_COUNT}/${FAIL_THRESHOLD}"
  fi

  do_restart_if_needed "missing process(es): ${missing_proc[*]}"
  exit 0
fi

# Restart sonrası tekrar ayağa kalktı mesajı (1 kez)
if (( LAST_RESTART > 0 )) && (( LAST_RECOVERY_TS == 0 )); then
  snap="$(stream_snapshot)"
  tg_ok "✅ WATCHDOG OK: binance1-orch recovered after restart\n${snap}\nts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# -----------------------------
# Stream activity (OR logic)
# -----------------------------
sig1="$(last_id signals_stream)"
exe1="$(last_id exec_events_stream)"

# Redis/stream okunamıyorsa: fail say (boşlukları debug'lamak zor)
# Not: stream gerçekten boş olabilir; ama senin sistemde normalde akış var. Bu yüzden boşluğu problem sayıyoruz.
if [[ -z "${sig1:-}" && -z "${exe1:-}" ]]; then
  bump_fail
  do_restart_if_needed "redis/streams unreadable or empty (signals_stream and exec_events_stream)"
  exit 0
fi

sleep "$SLEEP_SEC"

sig2="$(last_id signals_stream)"
exe2="$(last_id exec_events_stream)"

moved_sig=0
moved_exe=0

if [[ -n "${sig1:-}" && -n "${sig2:-}" && "${sig2}" != "${sig1}" ]]; then
  moved_sig=1
fi

if [[ -n "${exe1:-}" && -n "${exe2:-}" && "${exe2}" != "${exe1}" ]]; then
  moved_exe=1
fi

if (( moved_sig == 1 || moved_exe == 1 )); then
  echo "[WD][OK] stream activity: signals ${sig1:-na} -> ${sig2:-na}, exec ${exe1:-na} -> ${exe2:-na}"
  reset_fail
  exit 0
fi

# No movement in window: allow if recent activity exists (either stream)
nm="$(now_ms)"
sig2m="$(id_ms "$sig2")"
exe2m="$(id_ms "$exe2")"

age_sig="na"
age_exe="na"

if [[ -n "${sig2m:-}" ]]; then
  age_sig=$(( (nm - sig2m) / 1000 ))
fi

if [[ -n "${exe2m:-}" ]]; then
  age_exe=$(( (nm - exe2m) / 1000 ))
fi

recent_ok=0
if [[ "$age_sig" != "na" ]] && (( age_sig <= MAX_IDLE_SEC )); then
  recent_ok=1
fi
if [[ "$age_exe" != "na" ]] && (( age_exe <= MAX_IDLE_SEC )); then
  recent_ok=1
fi

if [[ "$recent_ok" -eq 1 ]]; then
  echo "[WD][OK] no growth in ${SLEEP_SEC}s but recent activity (age_sig=${age_sig}s age_exec=${age_exe}s <= ${MAX_IDLE_SEC}s)"
  reset_fail
  exit 0
fi

bump_fail
do_restart_if_needed "no activity (>${MAX_IDLE_SEC}s) (age_sig=${age_sig}s age_exec=${age_exe}s)"
exit 0

