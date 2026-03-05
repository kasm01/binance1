#!/usr/bin/env bash
set -euo pipefail

# =========================
# binance1 Supervisor
# - keepalive: main.py
# - optional: manage orchestration systemd service (orch owns bridge)
# - Telegram alerts (rate-limited)
# - Redis stream group health: pending/lag thresholds
# - Redis stream self-healing: XAUTOCLAIM stuck pending
# - STREAM BACKPRESSURE GUARD: exec_events_stream XLEN>threshold -> throttle scanners via Redis key
# - log rotation
# =========================

BASE_DIR="${BASE_DIR:-$HOME/binance1}"
VENV_PY="$BASE_DIR/venv/bin/python"
ENV_FILE="$BASE_DIR/.env"

MAIN_CMD=("$VENV_PY" -u "$BASE_DIR/main.py")

# Logs
SUP_LOG_DIR="$BASE_DIR/logs/supervisor"
mkdir -p "$SUP_LOG_DIR"
SUP_LOG="$SUP_LOG_DIR/supervisor.log"

MAIN_LOG="$BASE_DIR/logs/main.log"

# Behavior
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-10}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-10485760}"   # 10MB
ROTATE_KEEP="${ROTATE_KEEP:-5}"

# Orchestration service
ORCH_SERVICE="${ORCH_SERVICE:-binance1-orch.service}"
MANAGE_ORCH_SERVICE="${MANAGE_ORCH_SERVICE:-1}"   # 1=yes, 0=no
# If orch manages intent_bridge, supervisor must NOT spawn it.
OWNS_ORCH_PROCS="${OWNS_ORCH_PROCS:-0}"           # 1=supervisor starts orch procs, 0=orch service does

# Safety defaults (override in .env)
export DRY_RUN="${DRY_RUN:-1}"
export ARMED="${ARMED:-0}"
export LIVE_KILL_SWITCH="${LIVE_KILL_SWITCH:-1}"
export BINANCE_TESTNET="${BINANCE_TESTNET:-0}"

# Telegram alert knobs
SUP_TG_ENABLED="${SUP_TG_ENABLED:-1}"                 # 1=enable, 0=disable
SUP_TG_COOLDOWN_SEC="${SUP_TG_COOLDOWN_SEC:-300}"     # per alert key
SUP_TG_SILENT="${SUP_TG_SILENT:-1}"                   # 1=silent, 0=notify

# Group health knobs
SUP_GROUP_HEALTH_ENABLED="${SUP_GROUP_HEALTH_ENABLED:-1}"
SUP_PENDING_WARN="${SUP_PENDING_WARN:-200}"
SUP_LAG_WARN="${SUP_LAG_WARN:-200}"
# stream:group list
SUP_GROUP_CHECKS="${SUP_GROUP_CHECKS:-\
candidates_stream:topsel_g,\
top5_stream:master_exec_g,\
trade_intents_stream:bridge_g,\
exec_events_stream:main_exec_g\
}"

# -------------------------
# BACKPRESSURE GUARD knobs
# -------------------------
SUP_BACKPRESSURE_ENABLED="${SUP_BACKPRESSURE_ENABLED:-1}"

# Streams (can be overridden by .env)
EXEC_EVENTS_STREAM="${EXEC_EVENTS_STREAM:-exec_events_stream}"

# XLEN thresholds
SUP_BP_HIGH_WATER="${SUP_BP_HIGH_WATER:-5000}"     # if XLEN > this -> increase throttle
SUP_BP_LOW_WATER="${SUP_BP_LOW_WATER:-2500}"       # if XLEN < this -> relax throttle (hysteresis)

# throttle key + behavior
SUP_BP_THROTTLE_KEY="${SUP_BP_THROTTLE_KEY:-scanner:throttle_factor}"
SUP_BP_TTL_SEC="${SUP_BP_TTL_SEC:-120}"            # refresh TTL while pressure exists
SUP_BP_MIN="${SUP_BP_MIN:-1}"
SUP_BP_MAX="${SUP_BP_MAX:-8}"
SUP_BP_STEP_UP="${SUP_BP_STEP_UP:-1}"
SUP_BP_STEP_DOWN="${SUP_BP_STEP_DOWN:-1}"

# How often guard runs (ticks)
SUP_BP_EVERY_TICKS="${SUP_BP_EVERY_TICKS:-1}"

# Health alert (cooldown’lu) keys
SUP_BP_ALERT_KEY="${SUP_BP_ALERT_KEY:-bp_exec_events}"
SUP_GROUP_ALERT_PREFIX="${SUP_GROUP_ALERT_PREFIX:-group_warn}"
SUP_REPAIR_ALERT_KEY="${SUP_REPAIR_ALERT_KEY:-xautoclaim_repair}"

# -------------------------
# Utils
# -------------------------
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
TG_CD_FILE="$SUP_LOG_DIR/tg_cooldowns.json"

tg_cd_get() {
  local key="$1"
  [[ -f "$TG_CD_FILE" ]] || { echo "0"; return 0; }
  python3 - <<PY 2>/dev/null || echo "0"
import json
p="$TG_CD_FILE"; k="$key"
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
import json, os
p="$TG_CD_FILE"; k="$key"; now=int("$now_ts")
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

  local token="${TELEGRAM_BOT_TOKEN:-}"
  local chat_id="${TELEGRAM_CHAT_ID:-}"

  if [[ -z "$token" || -z "$chat_id" || "$token" == "SET" || "$chat_id" == "SET" ]]; then
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

  curl -sS -X POST "https://api.telegram.org/bot${token}/sendMessage" \
    -d "chat_id=${chat_id}" \
    --data-urlencode "text=${text}" \
    -d "disable_notification=${silent}" >/dev/null 2>&1 || true
}

# -------------------------
# Process control
# -------------------------
is_running_main() {
  pgrep -af "python.*$BASE_DIR/main\.py" >/dev/null 2>&1
}

start_main() {
  if is_running_main; then
    return 0
  fi
  log "starting main.py..."
  tg_send "main_start" "🟡 binance1: main.py starting (dry_run=${DRY_RUN}, kill=${LIVE_KILL_SWITCH})"
  nohup "${MAIN_CMD[@]}" >>"$MAIN_LOG" 2>&1 &
  echo $! > "$SUP_LOG_DIR/main.pid" 2>/dev/null || true
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

orch_state() {
  if [[ "$MANAGE_ORCH_SERVICE" != "1" ]]; then
    echo "disabled"
    return 0
  fi
  local st
  st="$(systemctl --user is-active "$ORCH_SERVICE" 2>/dev/null || echo "unknown")"
  echo "$st"
}

# -------------------------
# Group health check (XINFO GROUPS)
# -------------------------
group_stats() {
  # prints: pending lag (or: -1 -1 if not found)
  local stream="$1"
  local group="$2"
  local db="${REDIS_DB:-0}"

  redis-cli -n "$db" --raw XINFO GROUPS "$stream" 2>/dev/null | awk -v g="$group" '
    BEGIN{found=0; pending=-1; lag=-1;}
    $0=="name" {getline; if($0==g){found=1}else{found=0}}
    found==1 && $0=="pending" {getline; pending=$0}
    found==1 && $0=="lag" {getline; lag=$0}
    END{print pending, lag}
  ' | tr -d '\r'
}

check_group_health() {
  [[ "$SUP_GROUP_HEALTH_ENABLED" == "1" ]] || return 0
  [[ -n "${SUP_GROUP_CHECKS:-}" ]] || return 0
  redis_ok || return 0

  local db="${REDIS_DB:-0}"
  local IFS=','

  for pair in $SUP_GROUP_CHECKS; do
    local stream="${pair%%:*}"
    local group="${pair#*:}"
    [[ -n "$stream" && -n "$group" ]] || continue

    local stats pending lag
    stats="$(group_stats "$stream" "$group" || echo "-1 -1")"
    pending="$(echo "$stats" | awk '{print $1}')"
    lag="$(echo "$stats" | awk '{print $2}')"

    if [[ "$pending" == "-1" || "$lag" == "-1" ]]; then
      log "WARN: group missing/unknown: ${stream}:${group}"
      tg_send "${SUP_GROUP_ALERT_PREFIX}_missing_${stream}_${group}" \
        "⚠️ binance1: group missing/unknown: ${stream}:${group}"
      continue
    fi

    if [[ "$pending" -ge "$SUP_PENDING_WARN" || "$lag" -ge "$SUP_LAG_WARN" ]]; then
      log "WARN: group unhealthy: ${stream}:${group} pending=$pending lag=$lag (th=${SUP_PENDING_WARN}/${SUP_LAG_WARN})"
      tg_send "${SUP_GROUP_ALERT_PREFIX}_${stream}_${group}" \
        "⚠️ binance1: group warn ${stream}:${group} pending=${pending} lag=${lag} (th=${SUP_PENDING_WARN}/${SUP_LAG_WARN})"
    fi
  done
}

# -------------------------
# XAUTOCLAIM pending self-heal
# -------------------------
recover_pending() {
  [[ "$SUP_GROUP_HEALTH_ENABLED" == "1" ]] || return 0
  [[ -n "${SUP_GROUP_CHECKS:-}" ]] || return 0
  redis_ok || return 0

  local db="${REDIS_DB:-0}"
  local IFS=','

  local total_claimed=0

  for pair in $SUP_GROUP_CHECKS; do
    local stream="${pair%%:*}"
    local group="${pair#*:}"
    [[ -n "$stream" && -n "$group" ]] || continue

    # idle > 60s pending msgleri claim et (best-effort)
    # reply format: [next_start_id, [ [id, [k,v...]], ... ], [deleted_ids...]]
    local out
    out="$(redis-cli -n "$db" XAUTOCLAIM "$stream" "$group" "supervisor_recover" 60000 0 COUNT 50 2>/dev/null || true)"
    if [[ -n "${out//[[:space:]]/}" ]]; then
      # crude estimate: count IDs in output (lines that look like stream ids)
      local claimed
      claimed="$(printf "%s\n" "$out" | grep -E '^[0-9]{13,}-[0-9]+$' | wc -l | tr -d ' ')"
      if [[ "${claimed:-0}" -gt 0 ]]; then
        total_claimed=$((total_claimed + claimed))
      fi
    fi
  done

  if [[ "$total_claimed" -gt 0 ]]; then
    log "INFO: XAUTOCLAIM repaired pending msgs total_claimed=$total_claimed"
    tg_send "$SUP_REPAIR_ALERT_KEY" "🛠️ binance1: XAUTOCLAIM repaired pending msgs (claimed=${total_claimed})"
  fi
}

# -------------------------
# BACKPRESSURE GUARD
# -------------------------
redis_get_int() {
  local key="$1"
  local db="${REDIS_DB:-0}"
  local v
  v="$(redis-cli -n "$db" GET "$key" 2>/dev/null | tr -d '\r' || true)"
  [[ -z "${v:-}" ]] && { echo ""; return 0; }
  # if quoted like "2"
  v="${v%\"}"; v="${v#\"}"
  echo "$v"
}

redis_set_throttle() {
  local factor="$1"
  local ttl="$2"
  local db="${REDIS_DB:-0}"
  redis-cli -n "$db" SET "$SUP_BP_THROTTLE_KEY" "$factor" EX "$ttl" >/dev/null 2>&1 || true
}

backpressure_guard() {
  [[ "$SUP_BACKPRESSURE_ENABLED" == "1" ]] || return 0
  redis_ok || return 0

  local db="${REDIS_DB:-0}"
  local xlen
  xlen="$(redis-cli -n "$db" XLEN "$EXEC_EVENTS_STREAM" 2>/dev/null | tr -d '\r' || echo "0")"

  # current throttle (default=1)
  local cur
  cur="$(redis_get_int "$SUP_BP_THROTTLE_KEY")"
  if [[ -z "${cur:-}" ]]; then cur="$SUP_BP_MIN"; fi
  if ! [[ "$cur" =~ ^[0-9]+$ ]]; then cur="$SUP_BP_MIN"; fi

  local next="$cur"
  local action="hold"

  if [[ "$xlen" -gt "$SUP_BP_HIGH_WATER" ]]; then
    next=$((cur + SUP_BP_STEP_UP))
    action="up"
  elif [[ "$xlen" -lt "$SUP_BP_LOW_WATER" ]]; then
    next=$((cur - SUP_BP_STEP_DOWN))
    action="down"
  else
    action="hold"
  fi

  # clamp
  if [[ "$next" -lt "$SUP_BP_MIN" ]]; then next="$SUP_BP_MIN"; fi
  if [[ "$next" -gt "$SUP_BP_MAX" ]]; then next="$SUP_BP_MAX"; fi

  # refresh TTL while pressure exists OR while throttle > 1
  # (so it doesn't disappear mid-recovery)
  if [[ "$xlen" -gt "$SUP_BP_LOW_WATER" || "$next" -gt 1 ]]; then
    redis_set_throttle "$next" "$SUP_BP_TTL_SEC"
  else
    # if fully healthy & next==1, allow key to expire naturally (don't refresh)
    :
  fi

  if [[ "$action" == "up" && "$next" -gt "$cur" ]]; then
    log "WARN: BACKPRESSURE exec_events_stream XLEN=$xlen > $SUP_BP_HIGH_WATER -> throttle $cur -> $next (ttl=${SUP_BP_TTL_SEC}s)"
    tg_send "$SUP_BP_ALERT_KEY" "🚨 binance1 BACKPRESSURE: XLEN(${EXEC_EVENTS_STREAM})=${xlen} (> ${SUP_BP_HIGH_WATER}) throttle=${next} (ttl=${SUP_BP_TTL_SEC}s)"
  elif [[ "$action" == "down" && "$next" -lt "$cur" ]]; then
    log "INFO: BACKPRESSURE relax XLEN=$xlen < $SUP_BP_LOW_WATER -> throttle $cur -> $next"
    tg_send "${SUP_BP_ALERT_KEY}_relax" "✅ binance1 BACKPRESSURE relaxing: XLEN(${EXEC_EVENTS_STREAM})=${xlen} (< ${SUP_BP_LOW_WATER}) throttle=${next}"
  fi
}

health_report() {
  local db="${REDIS_DB:-0}"
  local main="down" redis="down"
  local orch="$(orch_state)"

  is_running_main && main="up"
  redis_ok && redis="up"

  local xlen_exec="?"
  xlen_exec="$(redis-cli -n "$db" XLEN "$EXEC_EVENTS_STREAM" 2>/dev/null || echo "?")"

  local thr="(none)"
  local thr_v
  thr_v="$(redis_get_int "$SUP_BP_THROTTLE_KEY")"
  [[ -n "${thr_v:-}" ]] && thr="$thr_v"

  log "health: redis=$redis orch=$orch main=$main XLEN(exec)=$xlen_exec throttle=$thr"
}
# -------------------------
# Shutdown handling (clean exit so systemd stop won't timeout)
# -------------------------
SHUTDOWN_REQ=0
_on_term() {
  SHUTDOWN_REQ=1
  log "SIGTERM/INT received -> shutdown requested"
}
trap _on_term SIGTERM SIGINT

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

  log "supervisor started. base=$BASE_DIR dry_run=$DRY_RUN armed=$ARMED kill=$LIVE_KILL_SWITCH manage_orch=$MANAGE_ORCH_SERVICE owns_orch_procs=$OWNS_ORCH_PROCS"
  tg_send "sup_start" "🟢 binance1: supervisor started (dry_run=${DRY_RUN}, kill=${LIVE_KILL_SWITCH})"

  ensure_orch_service

  start_main
  health_report

  local tick=0

  while true; do
    tick=$((tick+1))

    if [[ "$SHUTDOWN_REQ" == "1" ]]; then
      log "shutdown flag set -> exiting supervisor loop"
      break
    fi

    rotate_log "$SUP_LOG"
    rotate_log "$MAIN_LOG"

    if ! redis_ok; then
      log "WARN: Redis ping failed. will retry."
      tg_send "redis_down" "⚠️ binance1: Redis PING failed (will retry)"
      sleep "$CHECK_EVERY_SEC"
      continue
    fi

    ensure_orch_service

    if ! is_running_main; then
      log "main down -> restarting"
      tg_send "main_down" "🔴 binance1: main.py DOWN -> restarting"
      start_main
    fi

    # ---- BACKPRESSURE GUARD (periyodik) ----
    if [[ $((tick % SUP_BP_EVERY_TICKS)) -eq 0 ]]; then
      backpressure_guard
    fi

    # ---- GROUP HEALTH + SELF-HEAL (every 3 ticks) ----
    if [[ $((tick % 3)) -eq 0 ]]; then
      check_group_health
      recover_pending
    fi

    # ---- periodic report (every 6 ticks) ----
    if [[ $((tick % 6)) -eq 0 ]]; then
      health_report
    fi

    sleep "$CHECK_EVERY_SEC"
  done

  log "supervisor exiting cleanly"
}

main_loop
