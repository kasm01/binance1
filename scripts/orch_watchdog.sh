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
MAX_IDLE_SEC="${WATCHDOG_MAX_IDLE_SEC:-15}"   # growth yoksa bile son event <=15s ise OK

# restart için 2 kez üst üste fail şartı
FAIL_THRESHOLD="${WATCHDOG_FAIL_THRESHOLD:-2}"

# restartlar arası minimum süre (sn)
COOLDOWN_SEC="${WATCHDOG_RESTART_COOLDOWN_SEC:-300}"  # 5 dk

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
  # atomic write
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
  redis-cli XREVRANGE "$1" + - COUNT 1 2>/dev/null | head -n 1 | tr -d '\r' || true
}

id_ms() {
  awk -F'-' '{print $1}' <<<"${1:-}" 2>/dev/null || true
}

# -----------------------------
# Start
# -----------------------------
read_state

# -----------------------------
# Process health (must exist)
# -----------------------------
must_patterns=(
  "orchestration/scanners/worker_stub.py"
  "orchestration/aggregator/run_aggregator.py"
  "orchestration/selector/top_selector.py"
  "orchestration/executor/master_executor.py"
  "orchestration/executor/intent_bridge.py"
)

missing_proc=()
for pat in "${must_patterns[@]}"; do
  if ! pgrep -af "$pat" >/dev/null 2>&1; then
    missing_proc+=("$pat")
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

sleep "$SLEEP_SEC"

sig2="$(last_id signals_stream)"
exe2="$(last_id exec_events_stream)"

# Either stream moved -> healthy
if [[ -n "${sig1:-}" && -n "${sig2:-}" && "$sig2" != "$sig1" ]] || \
   [[ -n "${exe1:-}" && -n "${exe2:-}" && "$exe2" != "$exe1" ]]; then
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
if [[ "$age_sig" != "na" ]] && (( age_sig <= MAX_IDLE_SEC )); then recent_ok=1; fi
if [[ "$age_exe" != "na" ]] && (( age_exe <= MAX_IDLE_SEC )); then recent_ok=1; fi

if [[ "$recent_ok" -eq 1 ]]; then
  echo "[WD][OK] no growth in ${SLEEP_SEC}s but recent activity (age_sig=${age_sig}s age_exec=${age_exe}s <= ${MAX_IDLE_SEC}s)"
  reset_fail
  exit 0
fi

# true fail
bump_fail
do_restart_if_needed "no activity (>${MAX_IDLE_SEC}s) (age_sig=${age_sig}s age_exec=${age_exe}s)"
exit 0

