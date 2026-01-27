#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

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

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/binance1"
STATE_FILE="$STATE_DIR/orch_watchdog.state"
mkdir -p "$STATE_DIR"

need() { command -v "$1" >/dev/null 2>&1 || { echo "[WD][FAIL] missing: $1"; exit 2; }; }
need redis-cli
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

now_ts() { date +%s; }
now_ms() { date +%s%3N; }

read_state() {
  FAIL_COUNT=0
  LAST_RESTART=0
  LAST_OK=0

  if [[ -f "${STATE_FILE:-}" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE" || true
  fi

  FAIL_COUNT="${FAIL_COUNT:-0}"
  LAST_RESTART="${LAST_RESTART:-0}"
  LAST_OK="${LAST_OK:-0}"
}

write_state() {
  local tmp="${STATE_FILE}.tmp"
  cat >"$tmp" <<EOF
FAIL_COUNT=$FAIL_COUNT
LAST_RESTART=$LAST_RESTART
LAST_OK=$LAST_OK
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

# -----------------------------
# Process health helpers (pidfile-aware + heal)
# -----------------------------
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

find_pid_by_expect() {
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
    newpid="$(find_pid_by_expect "$expect")"
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
    newpid="$(find_pid_by_expect "$expect")"
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
    newpid="$(find_pid_by_expect "$expect")"
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
  newpid="$(find_pid_by_expect "$expect")"
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
# Grace (restart sonrası false alarm keser)
# -----------------------------
ts_now="$(now_ts)"
if (( LAST_RESTART > 0 )) && (( ts_now - LAST_RESTART < GRACE_SEC )); then
  echo "[WD][OK] in grace period (age=$((ts_now - LAST_RESTART))s < ${GRACE_SEC}s)"
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
  do_restart_if_needed "missing process(es): ${missing_proc[*]}"
  exit 0
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

