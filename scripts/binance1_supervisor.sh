#!/usr/bin/env bash
set -euo pipefail

# =========================
# binance1 Supervisor
# - keepalive: main.py + intent_bridge
# - optional: keepalive orchestration systemd service
# - Telegram alerts (rate-limited)
# - Redis stream group health: pending/lag thresholds
# - log rotation
# =========================

BASE_DIR="${BASE_DIR:-$HOME/binance1}"
VENV_PY="$BASE_DIR/venv/bin/python"
ENV_FILE="$BASE_DIR/.env"

MAIN_CMD=("$VENV_PY" -u "$BASE_DIR/main.py")
BRIDGE_CMD=("$VENV_PY" -u "$BASE_DIR/orchestration/executor/intent_bridge.py")

# Logs
SUP_LOG_DIR="$BASE_DIR/logs/supervisor"
mkdir -p "$SUP_LOG_DIR"
SUP_LOG="$SUP_LOG_DIR/supervisor.log"

BRIDGE_LOG="$BASE_DIR/logs/orch/intent_bridge.log"
MAIN_LOG="$BASE_DIR/logs/main.log"

# Behavior
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-10}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-10485760}"   # 10MB
ROTATE_KEEP="${ROTATE_KEEP:-5}"

# Orchestration service
ORCH_SERVICE="${ORCH_SERVICE:-binance1-orch.service}"
MANAGE_ORCH_SERVICE="${MANAGE_ORCH_SERVICE:-1}"   # 1=yes, 0=no

# Safety defaults (override in .env)
export DRY_RUN="${DRY_RUN:-1}"
export ARMED="${ARMED:-0}"
export LIVE_KILL_SWITCH="${LIVE_KILL_SWITCH:-1}"
export BINANCE_TESTNET="${BINANCE_TESTNET:-0}"

# Telegram alert knobs
SUP_TG_ENABLED="${SUP_TG_ENABLED:-1}"                 # 1=enable, 0=disable
SUP_TG_COOLDOWN_SEC="${SUP_TG_COOLDOWN_SEC:-300}"     # rate limit per alert key
SUP_TG_SILENT="${SUP_TG_SILENT:-1}"                   # 1=silent, 0=notify

# Group health knobs
SUP_GROUP_HEALTH_ENABLED="${SUP_GROUP_HEALTH_ENABLED:-1}"
SUP_PENDING_WARN="${SUP_PENDING_WARN:-200}"           # pending threshold
SUP_LAG_WARN="${SUP_LAG_WARN:-200}"                   # lag threshold
# stream:group list (comma-separated)
# You can override in .env if your group names differ.
SUP_GROUP_CHECKS="${SUP_GROUP_CHECKS:-\
candidates_stream:topsel_g,\
top5_stream:master_exec_g,\
trade_intents_stream:bridge_g,\
exec_events_stream:main_exec_g\
}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$SUP_LOG"; }

load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

redis_ok() {
  local db="${REDIS_DB:-0}"
  redis-cli -n "$db" PING >/dev/null 2>&1
}

rotate_log() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local size
  size=$(wc -c <"$f" 2>/dev/null || echo 0)
  if [[ "$size" -lt "$LOG_MAX_BYTES" ]]; then
    return 0
  fi

  for ((i=ROTATE_KEEP-1; i>=1; i--)); do
    [[ -f "${f}.${i}" ]] && mv -f "${f}.${i}" "${f}.$((i+1))" || true
  done
  cp -f "$f" "${f}.1" || true
  : > "$f" || true
  log "log rotated: $f (size=$size)"
}

# -------------------------
# Telegram (rate-limited)
# -------------------------

# cooldown state stored in file (simple)
TG_CD_FILE="$SUP_LOG_DIR/tg_cooldowns.json"

tg_cd_get() {
  local key="$1"
  [[ -f "$TG_CD_FILE" ]] || { echo "0"; return 0; }
  python3 - <<PY 2>/dev/null || echo "0"
import json, time
p="$TG_CD_FILE"
k="$key"
try:
  obj=json.load(open(p,"r"))
except Exception:
  obj={}
print(int(obj.get(k,0)))
PY
}

tg_cd_set() {
  local key="$1"
  local now_ts
  now_ts="$(date +%s)"
  python3 - <<PY 2>/dev/null || true
import json, time, os
p="$TG_CD_FILE"
k="$key"
now=int("$now_ts")
try:
  obj=json.load(open(p,"r"))
except Exception:
  obj={}
obj[k]=now
tmp=p+".tmp"
with open(tmp,"w") as f:
  json.dump(obj,f,ensure_ascii=False)
os.replace(tmp,p)
PY
}

tg_send() {
  [[ "$SUP_TG_ENABLED" == "1" ]] || return 0

  # Env keys already exist in your project
  local token="${TELEGRAM_BOT_TOKEN:-}"
  local chat_id="${TELEGRAM_CHAT_ID:-}"

  if [[ -z "$token" || -z "$chat_id" || "$token" == "SET" || "$chat_id" == "SET" ]]; then
    # In your logs you print "SET", but actual env should be real values.
    # If not, just skip.
    return 0
  fi

  local key="$1"
  local text="$2"
  local now_s
  now_s="$(date +%s)"

  local last
  last="$(tg_cd_get "$key")"
  if [[ $((now_s - last)) -lt "$SUP_TG_COOLDOWN_SEC" ]]; then
    return 0
  fi

  tg_cd_set "$key"

  local silent="true"
  [[ "$SUP_TG_SILENT" == "0" ]] && silent="false"

  # no raw token in logs
  curl -sS -X POST "https://api.telegram.org/bot${token}/sendMessage" \
    -d "chat_id=${chat_id}" \
    --data-urlencode "text=${text}" \
    -d "disable_notification=${silent}" >/dev/null 2>&1 || true
}

# -------------------------
# Process control
# -------------------------

kill_duplicates() {
  # Ensure only one main / bridge is running
  pkill -f "python.*$BASE_DIR/main.py" >/dev/null 2>&1 || true
  pkill -f "python.*orchestration/executor/intent_bridge.py" >/dev/null 2>&1 || true
}

is_running_main() {
  pgrep -af "python.*$BASE_DIR/main.py" >/dev/null 2>&1
}

is_running_bridge() {
  pgrep -af "python.*orchestration/executor/intent_bridge.py" >/dev/null 2>&1
}

start_bridge() {
  mkdir -p "$(dirname "$BRIDGE_LOG")"
  if is_running_bridge; then
    return 0
  fi
  log "starting intent_bridge..."
  tg_send "bridge_start" "🟡 binance1: intent_bridge starting (dry_run=${DRY_RUN}, kill=${LIVE_KILL_SWITCH})"
  nohup "${BRIDGE_CMD[@]}" >>"$BRIDGE_LOG" 2>&1 &
  sleep 1
}

start_main() {
  if is_running_main; then
    return 0
  fi
  log "starting main.py..."
  tg_send "main_start" "🟡 binance1: main.py starting (dry_run=${DRY_RUN}, kill=${LIVE_KILL_SWITCH})"
  nohup "${MAIN_CMD[@]}" >>"$MAIN_LOG" 2>&1 &
  sleep 2
}

ensure_orch_service() {
  [[ "$MANAGE_ORCH_SERVICE" == "1" ]] || return 0
  systemctl --user is-active "$ORCH_SERVICE" >/dev/null 2>&1 || {
    log "orch service not active -> starting $ORCH_SERVICE"
    tg_send "orch_start" "🟡 binance1: $ORCH_SERVICE not active -> starting"
    systemctl --user start "$ORCH_SERVICE" >/dev/null 2>&1 || true
  }
}

# -------------------------
# Group health check
# -------------------------

group_stats() {
  # prints: pending lag  (or: -1 -1 if not found)
  local stream="$1"
  local group="$2"
  local db="${REDIS_DB:-0}"

  # XINFO GROUPS output is a flat list; parse with awk best-effort.
  redis-cli -n "$db" XINFO GROUPS "$stream" 2>/dev/null | awk -v g="$group" '
    BEGIN{found=0; pending=-1; lag=-1;}
    $0=="name" {getline; if($0==g){found=1}else{found=0}}
    found==1 && $0=="pending" {getline; pending=$0}
    found==1 && $0=="lag" {getline; lag=$0}
    END{print pending, lag}
  '
}

check_group_health() {
  [[ "$SUP_GROUP_HEALTH_ENABLED" == "1" ]] || return 0
  [[ "$SUP_GROUP_CHECKS" != "" ]] || return 0
  redis_ok || return 0

  local IFS=','

  for pair in $SUP_GROUP_CHECKS; do
    local stream="${pair%%:*}"
    local group="${pair#*:}"
    [[ -n "$stream" && -n "$group" ]] || continue

    local stats
    stats="$(group_stats "$stream" "$group" || echo "-1 -1")"

    local pending lag
    pending="$(echo "$stats" | awk '{print $1}')"
    lag="$(echo "$stats" | awk '{print $2}')"

    # if group missing, notify (rate-limited)
    if [[ "$pending" == "-1" || "$lag" == "-1" ]]; then
      log "WARN: group not found or XINFO failed: stream=$stream group=$group"
      tg_send "group_missing_${stream}_${group}" "⚠️ binance1: group missing/unknown: ${stream}:${group}"
      continue
    fi

    if [[ "$pending" -ge "$SUP_PENDING_WARN" || "$lag" -ge "$SUP_LAG_WARN" ]]; then
      log "WARN: group unhealthy: ${stream}:${group} pending=$pending lag=$lag"
      tg_send "group_warn_${stream}_${group}" "⚠️ binance1: group warn ${stream}:${group} pending=${pending} lag=${lag} (th=${SUP_PENDING_WARN}/${SUP_LAG_WARN})"
    fi
  done
}

health_report() {
  local db="${REDIS_DB:-0}"
  local exec_stream="${EXEC_EVENTS_STREAM:-exec_events_stream}"
  local in_stream="${TRADE_INTENTS_STREAM:-trade_intents_stream}"

  local main="down" bridge="down" redis="down"
  is_running_main && main="up"
  is_running_bridge && bridge="up"
  redis_ok && redis="up"

  local xlen_exec="?"
  local xlen_in="?"
  xlen_exec=$(redis-cli -n "$db" XLEN "$exec_stream" 2>/dev/null || echo "?")
  xlen_in=$(redis-cli -n "$db" XLEN "$in_stream" 2>/dev/null || echo "?")

  log "health: redis=$redis main=$main bridge=$bridge XLEN(exec)=$xlen_exec XLEN(intents)=$xlen_in"
}
recover_pending() {

  local db="${REDIS_DB:-0}"

  IFS=','

  for pair in $SUP_GROUP_CHECKS; do

    local stream="${pair%%:*}"
    local group="${pair#*:}"

    # idle > 60s olan pending mesajları claim et
    redis-cli -n "$db" \
      XAUTOCLAIM "$stream" "$group" "supervisor_recover" 60000 0 COUNT 20 \
      >/dev/null 2>&1 || true

  done
}
# -------------------------
# Main loop
# -------------------------

main_loop() {
  cd "$BASE_DIR" || exit 1
  load_env

  if [[ ! -x "$VENV_PY" ]]; then
    log "ERROR: venv python not found: $VENV_PY"
    tg_send "fatal_venv" "🛑 binance1: supervisor fatal - venv python missing: $VENV_PY"
    exit 1
  fi

  log "supervisor started. base=$BASE_DIR dry_run=$DRY_RUN armed=$ARMED kill=$LIVE_KILL_SWITCH"
  tg_send "sup_start" "🟢 binance1: supervisor started (dry_run=${DRY_RUN}, kill=${LIVE_KILL_SWITCH})"

  ensure_orch_service

  # Prevent Telegram polling conflict: kill duplicates then start clean
  kill_duplicates
  start_bridge
  start_main
  health_report

  local tick=0

  while true; do
    tick=$((tick+1))

    rotate_log "$SUP_LOG"
    rotate_log "$BRIDGE_LOG"
    rotate_log "$MAIN_LOG"

    if ! redis_ok; then
      log "WARN: Redis ping failed. will retry."
      tg_send "redis_down" "⚠️ binance1: Redis PING failed (will retry)"
      sleep "$CHECK_EVERY_SEC"
      continue
    fi

    ensure_orch_service

    if ! is_running_bridge; then
      log "bridge down -> restarting"
      tg_send "bridge_down" "🔴 binance1: intent_bridge DOWN -> restarting"
      start_bridge
    fi

    if ! is_running_main; then
      log "main down -> restarting"
      tg_send "main_down" "🔴 binance1: main.py DOWN -> restarting"
      start_main
    fi

    # group health (every 3 ticks by default)
    if [[ $((tick % 3)) -eq 0 ]]; then
      check_group_health

      # recover stuck pending messages
      recover_pending
    fi

    # periodic report (every 6 ticks)
    if [[ $((tick % 6)) -eq 0 ]]; then
      health_report
    fi

    sleep "$CHECK_EVERY_SEC"
  done
}
main_loop
