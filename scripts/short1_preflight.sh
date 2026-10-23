#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE_DIR:-$HOME/binance1}"
cd "$BASE"
set -a
source .env
[ -f .runtime_secrets.env ] && source .runtime_secrets.env || true
set +a
fail(){ echo "[FAIL] $*"; exit 1; }
ok(){ echo "[OK] $*"; }
warn(){ echo "[WARN] $*"; }

eqv(){ local k="$1"; local want="$2"; local got="${!k:-}"; [[ "$got" == "$want" ]] || fail "$k=$got expected=$want"; ok "$k=$got"; }

echo "=== SHORT1 V30 PREFLIGHT ==="
eqv MASTER_MIN_TRADE_SCORE 0.540
eqv MASTER_SHORT_SOFT_SCORE_ENABLE 1
eqv MASTER_SHORT_SOFT_SCORE_MIN 0.530
eqv MAX_OPEN_POSITIONS 1
eqv ROI_HARD_STOP_PCT -0.1200
eqv SHORT_DYNAMIC_STOP_ENABLE 1
eqv SHORT_DYNAMIC_STOP_CONFIRM_SEC 30
eqv SHORT_DYNAMIC_STOP_CATASTROPHIC_PCT 0.0600
eqv SHORT_STOP_REENTRY_COOLDOWN_SEC 900
eqv RISK_OI_CONFIRMED_BUILD_ENABLE 1
eqv RISK_OI_CONFIRMED_BUILD_MULTIPLIER 0.50
eqv MASTER_SHORT_MONEY_FLOW_ENABLE 1
eqv MASTER_SHORT_MONEY_FLOW_EXHAUSTION_60M -0.20

redis-cli PING | grep -qx PONG || fail "Redis down"
ok "Redis PONG"

key="context:oi:INJUSDT"
ttl="$(redis-cli TTL "$key")"
[[ "$ttl" =~ ^[0-9]+$ ]] && (( ttl > 0 )) || fail "OI context missing/stale ttl=$ttl"
ok "OI context ttl=${ttl}s"

/home/kasm920/venv/bin/python -m py_compile \
  orchestration/executor/master_executor.py \
  orchestration/context/market_context_contract.py \
  orchestration/context/entry_context_snapshot.py \
  orchestration/risk/risk_master.py \
  core/short_entry_candle_service.py \
  core/trade_executor_short.py
ok "core compile"

if grep -nE '^[[:space:]]*(SHORT_SCALE_IN_ENABLE|SHORT_MAX_ENTRY_LAYERS)=' .env >/dev/null; then
  fail "legacy layer env is active"
fi
ok "layers OFF / single-position config"

if [[ "${DRY_RUN:-1}" == "0" && "${ARMED:-0}" == "1" && "${LIVE_KILL_SWITCH:-1}" == "0" ]]; then
  /home/kasm920/venv/bin/python - <<'PYMODEL' || fail "promoted production INJUSDT 3m SHORT model missing/invalid"
from models.short_model_registry import ShortModelRegistry
m=ShortModelRegistry().get('INJUSDT','3m',require_promoted=True,require_execution_contract=True)
print('[OK] production SHORT model',m.meta.get('provider'),m.meta.get('promoted'))
PYMODEL
else
  if /home/kasm920/venv/bin/python - <<'PYMODEL' >/dev/null 2>&1
from models.short_model_registry import ShortModelRegistry
ShortModelRegistry().get('INJUSDT','3m',require_promoted=True,require_execution_contract=True)
PYMODEL
  then
    ok "production SHORT model present"
  else
    echo "[INFO] production SHORT model not promoted; live startup guard will remain fail-closed"
  fi
fi

oi_state="$(systemctl --user is-active binance1-oi-context.service || true)"
[[ "$oi_state" == active ]] || fail "OI collector state=$oi_state"
ok "OI collector active"

orch="$(systemctl --user is-active binance1-orch.service || true)"
sup="$(systemctl --user is-active binance1-supervisor.service || true)"
echo "[INFO] trade services orch=$orch supervisor=$sup"
echo "[INFO] DRY_RUN=${DRY_RUN:-unset} ARMED=${ARMED:-unset} LIVE_KILL_SWITCH=${LIVE_KILL_SWITCH:-unset}"

if [[ "${ARMED:-0}" == 0 && "${LIVE_KILL_SWITCH:-1}" == 1 ]]; then
  ok "system DISARMED"
else
  warn "system is not in disarmed state"
fi

PY=/home/kasm920/venv/bin/python
# V5 non-routing validation chain (Stage-4 strategy-specific units only).
for svc in binance1-short1-model-shadow.service binance1-short1-v30-shadow.service; do
  if systemctl --user is-active --quiet "$svc"; then echo "[OK] $svc active"; else echo "[FAIL] $svc inactive"; exit 31; fi
done
$PY - <<'PY'
import json,time,redis,sys
r=redis.Redis(host='localhost',port=6379,db=0,decode_responses=True)
expected='short_entry_quality_fresh_logistic_v5'
src='telemetry:short1:inj:model_shadow_v5'; life='telemetry:short1:v30:lifecycle_shadow_v5'
rows=r.xrevrange(src,count=1)
if rows:
 o=json.loads(rows[0][1].get('json','{}')); age=time.time()-float(o.get('ts') or 0); provider=str(o.get('model_provider') or '')
 if provider!=expected: print(f'[FAIL] V5 provider mismatch {provider}'); sys.exit(33)
 if not isinstance(o.get('entry_market_context'),dict): print('[FAIL] V5 entry context missing'); sys.exit(34)
 if age>360:
  print(f'[INFO] V5 source last candidate age={age:.1f}s; collector is active, no fresh candidate required')
 else:
  print(f'[OK] V5 source fresh age={age:.1f}s provider={provider}')
else:
 print('[INFO] V5 source has no event yet; collector active and awaiting candidate')
h=r.xrevrange(life,count=20); hb=None
for _,f in h:
 try:
  o=json.loads(f.get('json','{}'))
  if o.get('kind')=='HEARTBEAT': hb=o; break
 except Exception: pass
if not hb: print('[FAIL] V30 V5 heartbeat missing'); sys.exit(35)
age=time.time()-float(hb.get('ts') or 0)
if age>130: print(f'[FAIL] V30 V5 heartbeat stale age={age:.1f}s'); sys.exit(36)
print(f'[OK] V30 V5 heartbeat fresh age={age:.1f}s')
PY

echo "SHORT1_PREFLIGHT_OK"
