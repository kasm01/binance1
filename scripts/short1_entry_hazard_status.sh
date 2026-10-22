#!/usr/bin/env bash
set -u
cd /home/kasm920/binance1

echo '== ENTRY HAZARD SERVICES =='
for s in \
  binance1-short1-entry-hazard-v44.service \
  binance1-short1-entry-hazard-v43.service \
  binance1-short1-entry-hazard-shadow.service \
  binance1-short1-entry-hazard-outcome-ws.service
do
  printf '%-58s enabled=%-9s active=%s\n' "$s" \
    "$(systemctl --user is-enabled "$s" 2>/dev/null || true)" \
    "$(systemctl --user is-active "$s" 2>/dev/null || true)"
done

echo '== V44 STREAMS =='
/home/kasm920/venv/bin/python - <<'PY'
import redis,json,os
r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
streams=[
 ('V44 policy','telemetry:short1:inj:entry_hazard_orderbook_v44_shadow'),
 ('V44 outcome','telemetry:short1:inj:entry_hazard_orderbook_v44_outcomes'),
 ('V43 policy rollback','telemetry:short1:inj:entry_hazard_policy_v43_shadow'),
 ('V43 outcome rollback','telemetry:short1:inj:entry_hazard_policy_v43_outcomes_ws')]
for title,s in streams:
 print(title,'len=',r.xlen(s))
 for sid,f in r.xrevrange(s,count=2):
  try:o=json.loads(f.get('json','{}'))
  except:continue
  c=o.get('candidate') or {}
  if 'labels_from_30s_decision' in o:
   l5=(o.get('labels_from_30s_decision') or {}).get('5m') or {}
   print(' ',sid,o.get('policy_state'),'gate=',o.get('model_gate_shadow_state'),'5m=',l5.get('touch_40bp'),'mae=',l5.get('mae_pct'),'mfe=',l5.get('mfe_pct'))
  else:
   ob=o.get('orderbook_20s') or {}; af=o.get('absorption_features_20s') or {}
   print(' ',sid,o.get('stage'),o.get('state'),'gate=',c.get('model_gate_shadow_state'),'score=',round(float(c.get('score') or 0),5),
         'f20=',round(float((o.get('snapshot_20s') or {}).get('flow') or 0),3),'d5=',round(float(ob.get('depth5_imbalance') or 0),3),
         'micro=',round(float(ob.get('microprice_dev_bps') or 0),3),'eff=',af.get('down_bps_per_sell_flow'))
PY

echo '== PRODUCTION SAFETY =='
for s in binance1-orch.service binance1-supervisor.service; do printf '%-34s %s\n' "$s" "$(systemctl --user is-active "$s" 2>/dev/null || true)"; done
grep -E '^(DRY_RUN|ARMED|LIVE_KILL_SWITCH)=' .env || true

echo "== V44 RESEARCH SAMPLE =="
REPORT="$HOME/binance1/research/short/results/SHORT1_ORDERBOOK_ABSORPTION_V44.json"
if [[ -f "$REPORT" ]]; then
  /home/kasm920/venv/bin/python - "$REPORT" <<'PY'
import json,sys
p=sys.argv[1]
try:
    x=json.load(open(p))
    print('outcomes=',x.get('n'),'model_gate_eligible=',x.get('model_gate_eligible_n'),'actionable=',x.get('actionable_entry_n'),'days=',x.get('actionable_days'),'trace_actionable=',x.get('trace_actionable_n'),'book_reversal_watch=',x.get('book_reversal_watch_n'),'status=',x.get('sample_status'))
    br=x.get('book_reversal_watch_summary') or {}
    print('watch_5m=',br.get('touch_counts'),'watch_mae_med=',br.get('mae5_median_pct'),'watch_mfe_med=',br.get('mfe5_median_pct'))
except Exception as e:
    print('report_error=',e)
PY
else
  echo "report=missing"
fi
