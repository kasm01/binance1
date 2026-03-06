#!/usr/bin/env bash
set -euo pipefail

# =========================
# binance1 Supervisor (stable)
# - keepalive: main.py
# - optional: ensure orch service
# - Telegram alerts (rate-limited + normalized)
# - Group health (XINFO GROUPS) -> warn/recover (state-change)
# - Self-heal (XAUTOCLAIM) optional
# - Backpressure guard: XLEN(exec_events_stream) -> scanner throttle factor
# - XTRIM retention
# - Optional stuck reset policy
# - Reconcile helper (orch state <-> executor state)
# =========================

BASE_DIR="${BASE_DIR:-$HOME/binance1}"
ENV_FILE="${ENV_FILE:-$BASE_DIR/.env}"
VENV_PY="${VENV_PY:-$BASE_DIR/venv/bin/python}"
MAIN_CMD=("$VENV_PY" -u "$BASE_DIR/main.py")

# Logs
SUP_LOG_DIR="$BASE_DIR/logs/supervisor"
mkdir -p "$SUP_LOG_DIR"
SUP_LOG="$SUP_LOG_DIR/supervisor.log"
MAIN_LOG="$BASE_DIR/logs/main.log"
RECON_LOG="$SUP_LOG_DIR/reconcile.log"

CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-10}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-10485760}"   # 10MB
ROTATE_KEEP="${ROTATE_KEEP:-5}"

# Services
ORCH_SERVICE="${ORCH_SERVICE:-binance1-orch.service}"
MANAGE_ORCH_SERVICE="${MANAGE_ORCH_SERVICE:-1}"

# Safety defaults
export DRY_RUN="${DRY_RUN:-1}"
export ARMED="${ARMED:-0}"
export LIVE_KILL_SWITCH="${LIVE_KILL_SWITCH:-1}"
export BINANCE_TESTNET="${BINANCE_TESTNET:-0}"

# Redis
REDIS_DB="${REDIS_DB:-0}"

# Telegram
SUP_TG_ENABLED="${SUP_TG_ENABLED:-1}"
SUP_TG_COOLDOWN_SEC="${SUP_TG_COOLDOWN_SEC:-300}"
SUP_TG_SILENT="${SUP_TG_SILENT:-1}"
SUP_TG_HEALTH_MODE="${SUP_TG_HEALTH_MODE:-change}"         # change|periodic|off
SUP_TG_HEALTH_MIN_INTERVAL_SEC="${SUP_TG_HEALTH_MIN_INTERVAL_SEC:-1800}"

# Group health
SUP_GROUP_HEALTH_ENABLED="${SUP_GROUP_HEALTH_ENABLED:-1}"
SUP_PENDING_WARN="${SUP_PENDING_WARN:-200}"
SUP_LAG_WARN="${SUP_LAG_WARN:-200}"
SUP_GROUP_CHECKS="${SUP_GROUP_CHECKS:-candidates_stream:topsel_g,top5_stream:master_exec_g,trade_intents_stream:bridge_g,exec_events_stream:main_exec_g}"

# XAUTOCLAIM
SUP_XAUTOCLAIM_ENABLED="${SUP_XAUTOCLAIM_ENABLED:-1}"
SUP_XAUTOCLAIM_MIN_IDLE_MS="${SUP_XAUTOCLAIM_MIN_IDLE_MS:-60000}"
SUP_XAUTOCLAIM_COUNT="${SUP_XAUTOCLAIM_COUNT:-50}"
SUP_XAUTOCLAIM_EVERY_TICKS="${SUP_XAUTOCLAIM_EVERY_TICKS:-3}"

# Stuck reset policy
SUP_STUCK_RESET_MODE="${SUP_STUCK_RESET_MODE:-none}"       # none|setid_tail
SUP_STUCK_CONSEC_TICKS="${SUP_STUCK_CONSEC_TICKS:-6}"
SUP_STUCK_RESET_EVERY_TICKS="${SUP_STUCK_RESET_EVERY_TICKS:-3}"

# Backpressure guard
SUP_BACKPRESSURE_ENABLED="${SUP_BACKPRESSURE_ENABLED:-1}"
EXEC_EVENTS_STREAM="${EXEC_EVENTS_STREAM:-exec_events_stream}"
SUP_BP_HIGH_WATER="${SUP_BP_HIGH_WATER:-5000}"
SUP_BP_LOW_WATER="${SUP_BP_LOW_WATER:-2500}"
SUP_BP_THROTTLE_KEY="${SUP_BP_THROTTLE_KEY:-scanner:throttle_factor}"
SUP_BP_TTL_SEC="${SUP_BP_TTL_SEC:-120}"
SUP_BP_MIN="${SUP_BP_MIN:-1}"
SUP_BP_MAX="${SUP_BP_MAX:-8}"
SUP_BP_STEP_UP="${SUP_BP_STEP_UP:-1}"
SUP_BP_STEP_DOWN="${SUP_BP_STEP_DOWN:-1}"
SUP_BP_EVERY_TICKS="${SUP_BP_EVERY_TICKS:-1}"

# XTRIM retention
SUP_TRIM_ENABLED="${SUP_TRIM_ENABLED:-1}"
SUP_TRIM_MAXLEN="${SUP_TRIM_MAXLEN:-50000}"
SUP_TRIM_EVERY_TICKS="${SUP_TRIM_EVERY_TICKS:-6}"
SUP_TRIM_STREAMS="${SUP_TRIM_STREAMS:-signals_stream,candidates_stream,top5_stream,trade_intents_stream,exec_events_stream}"

# Reconcile
RUN_RECONCILE_ON_START="${RUN_RECONCILE_ON_START:-1}"
RUN_RECONCILE_DRY_EVERY_TICKS="${RUN_RECONCILE_DRY_EVERY_TICKS:-18}"
RECONCILE_SCRIPT="${RECONCILE_SCRIPT:-$BASE_DIR/scripts/reconcile_positions.py}"

# ------------ utils ------------
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$SUP_LOG" >/dev/null; }

need() { command -v "$1" >/dev/null 2>&1; }

load_env() {
  [[ -f "$ENV_FILE" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

rotate_log() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local size
  size=$(wc -c <"$f" 2>/dev/null || echo 0)
  [[ "$size" -lt "$LOG_MAX_BYTES" ]] && return 0
  for ((i=ROTATE_KEEP-1; i>=1; i--)); do
    [[ -f "${f}.${i}" ]] && mv -f "${f}.${i}" "${f}.$((i+1))" || true
  done
  cp -f "$f" "${f}.1" || true
  : > "$f" || true
  log "log rotated: $f (size=$size)"
}

redis_ok() {
  need redis-cli || return 1
  redis-cli -n "$REDIS_DB" PING >/dev/null 2>&1
}

redis_get_int() {
  local key="$1"
  local v
  v="$(redis-cli -n "$REDIS_DB" GET "$key" 2>/dev/null | tr -d '\r' || true)"
  v="${v%\"}"; v="${v#\"}"
  [[ -z "${v:-}" ]] && { echo ""; return 0; }
  [[ "$v" =~ ^[0-9]+$ ]] || { echo ""; return 0; }
  echo "$v"
}

redis_set_throttle() {
  local factor="$1" ttl="$2"
  redis-cli -n "$REDIS_DB" SET "$SUP_BP_THROTTLE_KEY" "$factor" EX "$ttl" >/dev/null 2>&1 || true
}

orch_state() {
  [[ "$MANAGE_ORCH_SERVICE" == "1" ]] || { echo "disabled"; return 0; }
  systemctl --user is-active "$ORCH_SERVICE" 2>/dev/null || echo "unknown"
}

ensure_orch_service() {
  [[ "$MANAGE_ORCH_SERVICE" == "1" ]] || return 0
  systemctl --user is-active "$ORCH_SERVICE" >/dev/null 2>&1 && return 0
  log "WARN orch not active -> starting $ORCH_SERVICE"
  systemctl --user start "$ORCH_SERVICE" >/dev/null 2>&1 || true
}

is_running_main() { pgrep -af "python.*$BASE_DIR/main\.py" >/dev/null 2>&1; }

start_main() {
  is_running_main && return 0
  log "INFO starting main.py"
  nohup "${MAIN_CMD[@]}" >>"$MAIN_LOG" 2>&1 &
  echo $! > "$SUP_LOG_DIR/main.pid" 2>/dev/null || true
  sleep 1
}

run_reconcile_apply() {
  [[ "$RUN_RECONCILE_ON_START" == "1" ]] || return 0
  [[ -x "$VENV_PY" ]] || return 0
  [[ -f "$RECONCILE_SCRIPT" ]] || return 0

  log "INFO reconcile apply start"
  "$VENV_PY" "$RECONCILE_SCRIPT" --apply --apply-exec >>"$RECON_LOG" 2>&1 || true
  log "INFO reconcile apply done"
}

run_reconcile_dry() {
  [[ "$RUN_RECONCILE_DRY_EVERY_TICKS" =~ ^[0-9]+$ ]] || return 0
  [[ "$RUN_RECONCILE_DRY_EVERY_TICKS" -gt 0 ]] || return 0
  [[ -x "$VENV_PY" ]] || return 0
  [[ -f "$RECONCILE_SCRIPT" ]] || return 0

  "$VENV_PY" "$RECONCILE_SCRIPT" >>"$RECON_LOG" 2>&1 || true
}

# ------------ telegram ------------
TG_CD_FILE="$SUP_LOG_DIR/tg_cooldowns.json"

tg_cd_get() {
  local key="$1"
  [[ -f "$TG_CD_FILE" ]] || { echo "0"; return 0; }
  python3 - <<PY 2>/dev/null || echo "0"
import json
p="$TG_CD_FILE"; k="$key"
try: obj=json.load(open(p,"r"))
except Exception: obj={}
print(int(obj.get(k,0)))
PY
}

tg_cd_set() {
  local key="$1"; local now_ts; now_ts="$(date +%s)"
  python3 - <<PY 2>/dev/null || true
import json, os
p="$TG_CD_FILE"; k="$key"; now=int("$now_ts")
try: obj=json.load(open(p,"r"))
except Exception: obj={}
obj[k]=now
tmp=p+".tmp"
with open(tmp,"w") as f: json.dump(obj,f,ensure_ascii=False)
os.replace(tmp,p)
PY
}

tg_send() {
  [[ "$SUP_TG_ENABLED" == "1" ]] || return 0
  need curl || return 0

  local token="${TELEGRAM_BOT_TOKEN:-}"
  local chat_id="${TELEGRAM_CHAT_ID:-}"
  [[ -n "$token" && -n "$chat_id" && "$token" != "SET" && "$chat_id" != "SET" ]] || return 0

  local key="$1" text="$2"
  local now_s last; now_s="$(date +%s)"; last="$(tg_cd_get "$key")"
  if [[ $((now_s - last)) -lt "$SUP_TG_COOLDOWN_SEC" ]]; then return 0; fi
  tg_cd_set "$key"

  local silent="true"; [[ "$SUP_TG_SILENT" == "0" ]] && silent="false"
  curl -sS -X POST "https://api.telegram.org/bot${token}/sendMessage" \
    -d "chat_id=${chat_id}" \
    --data-urlencode "text=${text}" \
    -d "disable_notification=${silent}" >/dev/null 2>&1 || true
}

tg_line() {
  local lvl="$1"; shift
  echo "binance1 | ${lvl} | $(ts) | $*"
}
# ------------ state ------------
declare -A LAST_GROUP_STATE
declare -A STUCK_CNT
LAST_HEALTH_TG_TS=0
LAST_HEALTH_SIG=""
LAST_BP_STATE=""
LAST_THR_SENT=""

# ------------ group health ------------
group_stats() {
  local stream="$1" group="$2"
  redis-cli -n "$REDIS_DB" --raw XINFO GROUPS "$stream" 2>/dev/null | awk -v g="$group" '
    BEGIN{found=0; p=-1; l=-1;}
    $0=="name"{getline; found=($0==g)?1:0}
    found==1 && $0=="pending"{getline; p=$0}
    found==1 && $0=="lag"{getline; l=$0}
    END{print p, l}
  ' | tr -d "\r"
}

check_group_health() {
  [[ "$SUP_GROUP_HEALTH_ENABLED" == "1" ]] || return 0
  redis_ok || return 0

  local IFS=',' pair stream group stats pending lag key state
  for pair in $SUP_GROUP_CHECKS; do
    stream="${pair%%:*}"; group="${pair#*:}"
    [[ -n "$stream" && -n "$group" ]] || continue
    key="${stream}:${group}"

    stats="$(group_stats "$stream" "$group" || echo "-1 -1")"
    pending="$(awk '{print $1}' <<<"$stats")"
    lag="$(awk '{print $2}' <<<"$stats")"

    if [[ "$pending" == "-1" || "$lag" == "-1" ]]; then
      state="MISS"
    elif [[ "$pending" -ge "$SUP_PENDING_WARN" || "$lag" -ge "$SUP_LAG_WARN" ]]; then
      state="WARN"
    else
      state="OK"
    fi

    if [[ "${LAST_GROUP_STATE[$key]:-}" != "$state" ]]; then
      LAST_GROUP_STATE[$key]="$state"
      if [[ "$state" == "WARN" ]]; then
        log "WARN group: $key pending=$pending lag=$lag (th=${SUP_PENDING_WARN}/${SUP_LAG_WARN})"
        tg_send "group_${key}_warn" "$(tg_line "WARN" "group=$key pending=$pending lag=$lag th=${SUP_PENDING_WARN}/${SUP_LAG_WARN}")"
      elif [[ "$state" == "MISS" ]]; then
        log "WARN group missing: $key"
        tg_send "group_${key}_miss" "$(tg_line "WARN" "group_missing=$key")"
      else
        log "INFO group recovered: $key pending=$pending lag=$lag"
        tg_send "group_${key}_ok" "$(tg_line "OK" "group_ok=$key pending=$pending lag=$lag")"
      fi
    fi

    if [[ "$state" == "WARN" ]]; then
      STUCK_CNT[$key]=$(( ${STUCK_CNT[$key]:-0} + 1 ))
    else
      STUCK_CNT[$key]=0
    fi
  done
}

stuck_reset_policy() {
  [[ "$SUP_STUCK_RESET_MODE" != "none" ]] || return 0
  redis_ok || return 0

  local IFS=',' pair stream group key stats pending lag
  for pair in $SUP_GROUP_CHECKS; do
    stream="${pair%%:*}"; group="${pair#*:}"
    key="${stream}:${group}"

    [[ "${STUCK_CNT[$key]:-0}" -ge "$SUP_STUCK_CONSEC_TICKS" ]] || continue

    stats="$(group_stats "$stream" "$group" || echo "-1 -1")"
    pending="$(awk '{print $1}' <<<"$stats")"
    lag="$(awk '{print $2}' <<<"$stats")"

    if [[ "$pending" == "0" && "$lag" -ge "$SUP_LAG_WARN" ]]; then
      if [[ "$SUP_STUCK_RESET_MODE" == "setid_tail" ]]; then
        log "WARN stuck reset(setid_tail): $key pending=0 lag=$lag -> XGROUP SETID $"
        redis-cli -n "$REDIS_DB" XGROUP SETID "$stream" "$group" "$" >/dev/null 2>&1 || true
        tg_send "stuck_reset_${key}" "$(tg_line "WARN" "stuck_reset=setid_tail group=$key lag=$lag -> setid=$")"
      fi
      STUCK_CNT[$key]=0
    fi
  done
}

recover_pending() {
  [[ "$SUP_XAUTOCLAIM_ENABLED" == "1" ]] || return 0
  redis_ok || return 0

  local IFS=',' pair stream group stats pending lag out claimed total=0
  for pair in $SUP_GROUP_CHECKS; do
    stream="${pair%%:*}"
    group="${pair#*:}"

    # exec_events_stream için supervisor claim yapmaz
    if [[ "$stream" == "$EXEC_EVENTS_STREAM" ]]; then
      continue
    fi

    stats="$(group_stats "$stream" "$group" || echo "-1 -1")"
    pending="$(awk '{print $1}' <<<"$stats")"
    lag="$(awk '{print $2}' <<<"$stats")"

    [[ "$pending" =~ ^[0-9]+$ ]] || continue
    [[ "$pending" -gt 0 ]] || continue

    out="$(redis-cli -n "$REDIS_DB" XAUTOCLAIM "$stream" "$group" "supervisor_recover" \
      "$SUP_XAUTOCLAIM_MIN_IDLE_MS" 0 COUNT "$SUP_XAUTOCLAIM_COUNT" 2>/dev/null || true)"

    claimed="$(printf "%s\n" "$out" | grep -E '^[0-9]{13,}-[0-9]+$' | wc -l | tr -d ' ' || true)"
    [[ "${claimed:-0}" -gt 0 ]] && total=$((total + claimed))
  done

  if [[ "$total" -gt 0 ]]; then
    log "INFO XAUTOCLAIM repaired total_claimed=$total"
    tg_send "xautoclaim_repair" "$(tg_line "INFO" "xautoclaim_repaired claimed=$total")"
  fi
}
backpressure_guard() {
  [[ "$SUP_BACKPRESSURE_ENABLED" == "1" ]] || return 0
  redis_ok || return 0

  local xlen cur next action
  xlen="$(redis-cli -n "$REDIS_DB" XLEN "$EXEC_EVENTS_STREAM" 2>/dev/null | tr -d '\r' || echo 0)"
  [[ "$xlen" =~ ^[0-9]+$ ]] || xlen=0

  cur="$(redis_get_int "$SUP_BP_THROTTLE_KEY")"
  [[ -z "${cur:-}" ]] && cur="$SUP_BP_MIN"

  next="$cur"; action="hold"
  if [[ "$xlen" -gt "$SUP_BP_HIGH_WATER" ]]; then
    next=$((cur + SUP_BP_STEP_UP)); action="up"
  elif [[ "$xlen" -lt "$SUP_BP_LOW_WATER" ]]; then
    next=$((cur - SUP_BP_STEP_DOWN)); action="down"
  fi

  [[ "$next" -lt "$SUP_BP_MIN" ]] && next="$SUP_BP_MIN"
  [[ "$next" -gt "$SUP_BP_MAX" ]] && next="$SUP_BP_MAX"

  if [[ "$xlen" -gt "$SUP_BP_LOW_WATER" || "$next" -gt 1 ]]; then
    redis_set_throttle "$next" "$SUP_BP_TTL_SEC"
  fi

  if [[ "$next" != "$cur" if [[ "$next" != "$cur" ]]; thenif [[ "$next" != "$cur" ]]; then "$LAST_BP_STATE" != "$next" ]]; then
      LAST_BP_STATE="$next"
    if [[ "$action" == "up" ]]; then
      log "WARN BACKPRESSURE XLEN=$xlen > $SUP_BP_HIGH_WATER throttle $cur->$next ttl=${SUP_BP_TTL_SEC}s"
      if [[ "$LAST_THR_SENT" != "$next" ]]; then
        LAST_THR_SENT="$next"
        tg_send "bp_up" "$(tg_line "ALERT" "backpressure XLEN(${EXEC_EVENTS_STREAM})=$xlen hi=$SUP_BP_HIGH_WATER throttle=$next ttl=${SUP_BP_TTL_SEC}s")"
      fi
    else
      log "INFO BACKPRESSURE relax XLEN=$xlen < $SUP_BP_LOW_WATER throttle $cur->$next"
      tg_send "bp_down" "$(tg_line "OK" "backpressure_relax XLEN(${EXEC_EVENTS_STREAM})=$xlen lo=$SUP_BP_LOW_WATER throttle=$next")"
      [[ "$next" -le 1 ]] && LAST_THR_SENT="1"
    fi
  fi
}

trim_streams() {
  [[ "$SUP_TRIM_ENABLED" == "1" ]] || return 0
  redis_ok || return 0
  [[ "$SUP_TRIM_MAXLEN" =~ ^[0-9]+$ ]] || return 0

  local IFS=',' s
  for s in $SUP_TRIM_STREAMS; do
    s="$(echo "$s" | xargs)"
    [[ -z "$s" ]] && continue
    redis-cli -n "$REDIS_DB" XTRIM "$s" MAXLEN ~ "$SUP_TRIM_MAXLEN" >/dev/null 2>&1 || true
  done
}

health_report() {
  local redis="down" main="down" orch thr xlen bucket sig now
  redis_ok && redis="up"
  is_running_main && main="up"
  orch="$(orch_state)"

  xlen="$(redis-cli -n "$REDIS_DB" XLEN "$EXEC_EVENTS_STREAM" 2>/dev/null | tr -d '\r' || echo "?")"
  thr="$(redis_get_int "$SUP_BP_THROTTLE_KEY")"; [[ -z "${thr:-}" ]] && thr="$SUP_BP_MIN"

  bucket="ok"
  if [[ "$xlen" =~ ^[0-9]+$ ]]; then
    if [[ "$xlen" -gt "$SUP_BP_HIGH_WATER" ]]; then
      bucket="hi"
    elif [[ "$xlen" -lt "$SUP_BP_LOW_WATER" ]]; then
      bucket="lo"
    else
      bucket="mid"
    fi
  else
    bucket="na"
  fi

  log "health redis=$redis orch=$orch main=$main xlen_exec=$xlen throttle=$thr"

  [[ "$SUP_TG_HEALTH_MODE" == "off" ]] && return 0
  sig="${redis}|${orch}|${main}|${thr}|${bucket}"
  now="$(date +%s)"

  if [[ "$SUP_TG_HEALTH_MODE" == "change" ]]; then
    if [[ "$sig" != "$LAST_HEALTH_SIG" ]] && [[ $((now - LAST_HEALTH_TG_TS)) -ge "$SUP_TG_HEALTH_MIN_INTERVAL_SEC" ]]; then
LAST_BP_STATE=""
      LAST_HEALTH_SIG="$sig"
LAST_BP_STATE=""
      LAST_HEALTH_TG_TS="$now"
      tg_send "health" "$(tg_line "HEALTH" "redis=$redis orch=$orch main=$main xlen_exec=$xlen throttle=$thr")"
    fi
  else
    if [[ $((now - LAST_HEALTH_TG_TS)) -ge "$SUP_TG_HEALTH_MIN_INTERVAL_SEC" ]]; then
      LAST_HEALTH_TG_TS="$now"
      tg_send "health" "$(tg_line "HEALTH" "redis=$redis orch=$orch main=$main xlen_exec=$xlen throttle=$thr")"
    fi
  fi
}

# ------------ shutdown ------------
SHUTDOWN_REQ=0
_on_term() { SHUTDOWN_REQ=1; log "INFO SIGTERM/INT -> shutdown requested"; }
trap _on_term SIGTERM SIGINT

# ------------ main loop ------------
main_loop() {
  cd "$BASE_DIR" || exit 1
  load_env

  [[ -x "$VENV_PY" ]] || { log "ERROR venv python missing: $VENV_PY"; exit 1; }
  log "INFO supervisor started base=$BASE_DIR dry_run=$DRY_RUN armed=$ARMED kill=$LIVE_KILL_SWITCH manage_orch=$MANAGE_ORCH_SERVICE"

  ensure_orch_service
  run_reconcile_apply
  start_main
  health_report

  local tick=0
  while true; do
    tick=$((tick+1))
    [[ "$SHUTDOWN_REQ" == "1" ]] && break

    rotate_log "$SUP_LOG"
    rotate_log "$MAIN_LOG"
    rotate_log "$RECON_LOG"

    if ! redis_ok; then
      log "WARN redis ping failed (retry)"
      tg_send "redis_down" "$(tg_line "WARN" "redis_ping_failed retrying")"
      sleep "$CHECK_EVERY_SEC"
      continue
    fi

    ensure_orch_service

    if ! is_running_main; then
      log "WARN main.py down -> restart"
      tg_send "main_down" "$(tg_line "ALERT" "main.py DOWN -> restarting")"
      start_main
      run_reconcile_dry
    fi

    if [[ $((tick % SUP_BP_EVERY_TICKS)) -eq 0 ]]; then
      backpressure_guard
    fi

    if [[ $((tick % SUP_XAUTOCLAIM_EVERY_TICKS)) -eq 0 ]]; then
      check_group_health
      recover_pending
    fi

    if [[ $((tick % SUP_STUCK_RESET_EVERY_TICKS)) -eq 0 ]]; then
      stuck_reset_policy
    fi

    if [[ $((tick % SUP_TRIM_EVERY_TICKS)) -eq 0 ]]; then
      trim_streams
    fi

    if [[ "$RUN_RECONCILE_DRY_EVERY_TICKS" =~ ^[0-9]+$ ]] && [[ "$RUN_RECONCILE_DRY_EVERY_TICKS" -gt 0 ]]; then
      if [[ $((tick % RUN_RECONCILE_DRY_EVERY_TICKS)) -eq 0 ]]; then
        run_reconcile_dry
      fi
    fi

    if [[ $((tick % 6)) -eq 0 ]]; then
      health_report
    fi

    sleep "$CHECK_EVERY_SEC"
  done

  log "INFO supervisor exiting cleanly"
}

main_loop
