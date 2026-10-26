#!/usr/bin/env python3
from pathlib import Path
import json, os, re, subprocess, sys
ROOT=Path('/home/kasm920/binance1')
LONG=Path('/home/kasm920/binance1-pro')
ENV=ROOT/'.env'

def load_env(p):
    d={}
    for line in p.read_text().splitlines():
        s=line.strip()
        if not s or s.startswith('#') or '=' not in s: continue
        k,v=s.split('=',1); d[k.strip()]=v.strip().strip('"').strip("'")
    return d

def b(v,default=False):
    if v is None:return default
    return str(v).lower() in ('1','true','yes','on')

def result(name,ok,detail,blocking=True):
    status='PASS' if ok else ('BLOCK' if blocking else 'WARN')
    print(f'[{status}] {name}: {detail}')
    return ok or not blocking

env=load_env(ENV)
checks=[]
print('SHORT1 LIVE PREFLIGHT (read-only)')
print('--------------------------------')
# Live flags: safe now; target is documented, not modified.
checks.append(result('Current safety flags', env.get('DRY_RUN')=='1' and env.get('ARMED')=='0' and env.get('LIVE_KILL_SWITCH')=='1', f"DRY_RUN={env.get('DRY_RUN')} ARMED={env.get('ARMED')} LIVE_KILL_SWITCH={env.get('LIVE_KILL_SWITCH')}", blocking=False))
# Model artifacts
model=ROOT/'models/short_model_INJUSDT_3m_best.joblib'
meta=ROOT/'models/short_model_meta_INJUSDT_3m.json'
checks.append(result('Production SHORT model', model.exists() and meta.exists(), f'model={model.exists()} meta={meta.exists()}'))
# Promotion
promo=ROOT/'research/short/results/SHORT1_V30_LIVE_PROMOTION_GATE.json'
try: pd=json.loads(promo.read_text())
except Exception: pd={}
status=str(pd.get('promotion_status','MISSING'))
checks.append(result('Promotion gate', status in ('PRODUCTION','PROMOTE','APPROVED'), f"status={status} reasons={pd.get('reasons')}"))
# Hard/soft live policy
hard=float(env.get('MASTER_MIN_TRADE_SCORE','0') or 0)
soft_enable=b(env.get('MASTER_SHORT_SOFT_SCORE_ENABLE'),False)
checks.append(result('Hard score gate', hard>=0.555, f'hard={hard:.3f}'))
checks.append(result('Soft band live policy', not soft_enable, f'MASTER_SHORT_SOFT_SCORE_ENABLE={int(soft_enable)} (target live=0)'))
checks.append(result('Contra build veto', b(env.get('MASTER_CONTEXT_CONTRA_BUILD_REJECT'),False), f"value={env.get('MASTER_CONTEXT_CONTRA_BUILD_REJECT')}"))

# Hard-entry quality readiness (research-only evidence promoted into preflight)
quality_script=ROOT/'research/short/report_short1_hard_entry_quality.py'
try:
    cp=subprocess.run([str(ROOT/'venv/bin/python'), str(quality_script)], capture_output=True, text=True, timeout=30)
    qout=cp.stdout or ''
except Exception as exc:
    qout=f'ERROR {exc}'

def _metric(pattern, default=None):
    m=re.search(pattern, qout, re.M)
    return m.group(1) if m else default

episodes=int(_metric(r'^EPISODES=(\d+)', '0') or 0)
strong_n=int(_metric(r'^ STRONG: n=(\d+)', '0') or 0)
strong_win=float(_metric(r'^ STRONG: n=\d+ win60=([0-9.]+)%', '0') or 0.0)
watch_win=_metric(r'^ WATCH: n=\d+ win60=([0-9.]+)%', None)
weak_win=_metric(r'^ WEAK: n=\d+ win60=([0-9.]+)%', None)
quality_sample_ok = episodes >= 10 and strong_n >= 6
checks.append(result('Hard-entry quality sample', quality_sample_ok, f'episodes={episodes}/10 strong_episodes={strong_n}/6'))
quality_perf_ok = strong_n >= 6 and strong_win >= 65.0
checks.append(result('Hard-entry STRONG 60m quality', quality_perf_ok, f'strong_win60={strong_win:.1f}% target>=65%'))
if watch_win is not None or weak_win is not None:
    other_vals=[float(x) for x in (watch_win,weak_win) if x is not None]
    sep_ok = bool(other_vals) and strong_win > max(other_vals)
    checks.append(result('Hard-entry grade separation', sep_ok, f'strong={strong_win:.1f}% other_max={max(other_vals):.1f}%' if other_vals else 'insufficient comparison', blocking=False))
else:
    checks.append(result('Hard-entry grade separation', False, 'insufficient comparison', blocking=False))

# Central risk profile
rp=LONG/'orchestration/risk/central_risk_enforce_dry_profile.py'
txt=rp.read_text() if rp.exists() else ''
pa=bool(re.search(r'production_authorized\s*=\s*True',txt))
checks.append(result('Central risk production authority', pa, f'production_authorized={pa}'))
# Single-position defaults/config inspection
mtext=(ROOT/'orchestration/executor/master_executor.py').read_text()
itext=(ROOT/'orchestration/executor/intent_bridge.py').read_text()
# Report configured env if present; otherwise extract common defaults.
master_max=env.get('MASTER_MAX_POS') or env.get('MAX_POS') or env.get('MAX_OPEN_POSITIONS')
bridge_max=env.get('INTENT_MAX_OPEN') or env.get('MAX_OPEN') or env.get('MAX_OPEN_POSITIONS')
if master_max is None:
    mm=re.search(r'self\.max_pos\s*=\s*_env_int\([^\n]+\n?[^\n]*?([0-9]+)\s*\)',mtext)
    master_max=mm.group(1) if mm else 'default/unknown'
if bridge_max is None:
    bm=re.search(r'max_open[^\n]*_env_int\([^\n]+\n?[^\n]*?([0-9]+)\s*\)',itext,re.I)
    bridge_max=bm.group(1) if bm else 'default/unknown'
checks.append(result('Single-position semantic config', str(master_max)=='1' and str(bridge_max)=='1', f'master_max={master_max} bridge_max={bridge_max}', blocking=False))
# Research routing must not be relied upon in live.
checks.append(result('Research-only routing isolation', b(env.get('SHORT_DRYRUN_RESEARCH_MODEL_ENABLE'),False), 'research routing is enabled but code restricts it to dry/disarmed/killswitch', blocking=False))
print('--------------------------------')
blocking_fail=sum(1 for x in checks if not x)
if blocking_fail:
    print(f'READINESS=BLOCKED ({blocking_fail} blocking checks failed)')
    sys.exit(2)
print('READINESS=READY_FOR_CONTROLLED_ARM')
