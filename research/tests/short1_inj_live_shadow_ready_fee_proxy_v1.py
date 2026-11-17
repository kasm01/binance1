#!/usr/bin/env python3
"""Non-executable as-of markout for last ~24h V6 shadow READY events.

Uses public historical Binance 1m candle OPEN after observed signal time.
Net scenario: fee/slippage assumed, NOT actual fills or official fee schedule.
"""
import csv,json,math,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from statistics import mean
import requests,redis
from orchestration.shadow.short1_inj_shadow_signal_bus_v1 import recent_by_bar,process,V5,V6
root=Path('research/short/data')
# One bounded two-page download, reusable for later inspections.
today=datetime.now(timezone.utc)
end=(today.replace(second=0,microsecond=0)+timedelta(minutes=1))
start=end-timedelta(hours=26)
f=root/f'INJUSDT_1m_shadow_fee_markout_{today.strftime("%Y%m%d")}.csv'
if not f.exists():
 session=requests.Session();first=int(start.timestamp()*1000);last=int(end.timestamp()*1000)
 candles=[];cursor=first
 while cursor<last:
  r=session.get('https://fapi.binance.com/fapi/v1/klines',
   params={'symbol':'INJUSDT','interval':'1m','startTime':cursor,'endTime':last-1,'limit':1500},timeout=15)
  r.raise_for_status();p=r.json()
  if not isinstance(p,list) or not p:raise RuntimeError('no_public_1m_data')
  candles.extend(p);next_ms=int(p[-1][0])+60000
  if next_ms<=cursor:raise RuntimeError('no_page_progress')
  cursor=next_ms
  print('PUBLIC_BOUNDED_1M',len(p),'THRU',datetime.fromtimestamp(cursor/1000,timezone.utc).isoformat(),flush=True)
  time.sleep(.2)
 f.parent.mkdir(exist_ok=True,parents=True)
 with f.open('w',newline='') as g:
  w=csv.writer(g);w.writerow(['open_ms','open','close_ms','close'])
  for p in candles:w.writerow([p[0],p[1],p[6],p[4]])
else:print('REUSED',f)
with f.open() as g:
 market={int(v['open_ms']):(float(v['open']),float(v['close'])) for v in csv.DictReader(g)}
redis_client=redis.Redis(decode_responses=True)
v5=recent_by_bar(redis_client,V5,540)
v6=recent_by_bar(redis_client,V6,540)
matched=sorted(set(v5)&set(v6))
state={};ready=[]
for bar in matched:
 try:result,state=process(v6[bar],v5[bar],state,now_ms=bar+5000)
 except ValueError:continue
 if result['decision'] in ('SHADOW_READY_DIRECT','SHADOW_READY_FRESH_MODEL'):
  result['event_ts_ms']=int(float(v6[bar]['ts'])*1000)
  ready.append(result)
rows=[]
for e in ready:
 t=e['event_ts_ms']
 entry_ms=((t+59999)//60000)*60000
 if entry_ms not in market:continue
 entry=market[entry_ms][0]
 row=dict(intent_id=e['intent_id'],source_ms=e['completed_close_ms'],entry_ms=entry_ms,
          entry=entry,decision=e['decision'],v5_pass=e['v5_pass'],
          v6_score=e['v6_score'],confirmation=e['confirmation_state'])
 for h in (15,30,60):
  # Use exactly a minute bar that CLOSES at +h minutes; no future information at entry.
  exit_open=entry_ms+(h-1)*60000
  if exit_open not in market:
   row[f'price_short_{h}m_pct']=None
   row[f'net_roi3x_{h}m_fee08_slip06_pct']=None
  else:
   exit_px=market[exit_open][1]
   gross=(entry-exit_px)/entry*100
   row[f'price_short_{h}m_pct']=round(gross,5)
   # 0.08% roundtrip NOTIONAL fee + 0.06% roundtrip notional slippage;
   # 3x leverage, no funding; approximate % ROI on posted margin.
   row[f'net_roi3x_{h}m_fee08_slip06_pct']=round((gross-.08-.06)*3,5)
 rows.append(row)
resultfile=Path('research/results/SHORT1_INJ_LIVE_SHADOW_READY_FEE_SCENARIO_20261113.csv')
with resultfile.open('w',newline='') as g:
 fieldnames=list(rows[0]) if rows else ['intent_id']
 w=csv.DictWriter(g,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
print('READY_SHADOW_EVENTS',len(ready),'PRICE_MATCHED',len(rows))
for h in (15,30,60):
 net=[v[f'net_roi3x_{h}m_fee08_slip06_pct'] for v in rows if v.get(f'net_roi3x_{h}m_fee08_slip06_pct') is not None]
 gross=[v[f'price_short_{h}m_pct'] for v in rows if v.get(f'price_short_{h}m_pct') is not None]
 print('H',h,'N',len(net),'AVG_UNDERLYING_SHORT_%',round(mean(gross),4) if gross else None,
       'AVG_NET_3X_ROI_SCENARIO_%',round(mean(net),4) if net else None,
       'NET_WIN_COUNT',sum(x>0 for x in net))
print('ASSUMPTIONS_NOT_LIVE_PNL fee_roundtrip_notional=0.08% slippage_roundtrip_notional=0.06% leverage=3; no funding')
print('OUTPUT',resultfile)
