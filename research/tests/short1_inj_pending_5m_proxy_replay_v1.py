#!/usr/bin/env python3
"""Historical observational comparison; DOES NOT recreate native PENDING READY decisions.

INJUSDT 5m Binance candles downloaded for fixed Oct25–Nov2 range (no repeated worker).
Use only closed bars, classify future direction, not executable PnL.
"""
import csv,json,time
from datetime import datetime,timezone
from pathlib import Path
import requests,redis
root=Path(__file__).resolve().parents[2]
out=root/'research/results'
out.mkdir(parents=True,exist_ok=True)
file=root/'research/short/data/INJUSDT_5m_20261025_20261102_PENDING_REPLAY.csv'
start=int(datetime(2026,10,25,tzinfo=timezone.utc).timestamp()*1000)
end=int(datetime(2026,11,2,tzinfo=timezone.utc).timestamp()*1000)
if not file.is_file():
 print('PUBLIC_FUTURES_5M_DOWNLOAD_FIXED_WINDOW',start,end,flush=True)
 candles=[];cur=start;session=requests.Session()
 while cur<end:
  resp=session.get('https://fapi.binance.com/fapi/v1/klines',
   params={'symbol':'INJUSDT','interval':'5m','startTime':cur,'endTime':end-1,'limit':1500},
   timeout=15);resp.raise_for_status();rows=resp.json()
  if not isinstance(rows,list) or not rows:raise RuntimeError('historical_kline_empty')
  candles.extend(rows);nxt=int(rows[-1][0])+300000
  if nxt<=cur:raise RuntimeError('kline_page_no_progress')
  cur=nxt
  print('BARS_RECEIVED',len(rows),'LAST_UTC',datetime.fromtimestamp(int(rows[-1][0])/1000,timezone.utc).isoformat(),flush=True)
  time.sleep(.3)
 file.parent.mkdir(exist_ok=True,parents=True)
 with file.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(('open_ms','open','high','low','close','close_ms','quote_volume','taker_buy_quote'))
  for r in candles:w.writerow((r[0],r[1],r[2],r[3],r[4],r[6],r[7],r[10]))
else:print('REUSING_EXISTING_HISTORY',file)
with file.open() as f:
 prices=[dict(open_ms=int(z['open_ms']),close_ms=int(z['close_ms']),o=float(z['open']),c=float(z['close']),
                  h=float(z['high']),l=float(z['low'])) for z in csv.DictReader(f)]
assert all(prices[i+1]['open_ms']-prices[i]['open_ms']==300000 for i in range(len(prices)-1)),'missing 5m bars'
r=redis.Redis(decode_responses=True)
events=[]
for event_id,fields in r.xrange('exec_events_stream_short1'):
 try:pkg=json.loads(fields.get('json') or '{}')
 except:continue
 for x in pkg.get('items',[]):
  if x.get('symbol')!='INJUSDT' or x.get('side')!='short':continue
  score_state=str(x.get('model_score_gate_state') or '')
  if score_state not in ('READY','PENDING_CONTEXT'):continue
  ts=int(event_id.split('-')[0])
  if not start<=ts<end:continue
  events.append(dict(t=ts,id=str(x.get('intent_id')),model=x.get('model_provider'),
                     score=float(x.get('model_score') or 0),threshold=float(x.get('model_score_gate_threshold') or 0),
                     gate=score_state))
events.sort(key=lambda x:x['t'])
import bisect
closes=[x['close_ms'] for x in prices]
for x in events:
 idx=bisect.bisect_left(closes,x['t'])-1
 if idx<0:continue
 x['reference_close_ms']=prices[idx]['close_ms']
 x['reference_price']=prices[idx]['c']
 for mins in (15,30,60):
  k=idx+mins//5
  if k<len(prices):
   x[f'short_return_{mins}m_pct']=round(100*(x['reference_price']/prices[k]['c']-1),4)
  else:x[f'short_return_{mins}m_pct']=None
matched=[]
for i,e in enumerate(events):
 if e['gate']!='PENDING_CONTEXT':continue
 choices=[f for f in events[i+1:] if f['t']>e['t'] and f['t']<=e['t']+15*60*1000
          and f['gate']=='READY' and f['model']==e['model'] and f['id']!=e['id']
          and f.get('reference_close_ms',-1)>e.get('reference_close_ms',-2)]
 if choices:
  f=choices[0]
  matched.append(dict(pending_intent=e['id'],pending_ms=e['t'],pending_score=e['score'],
    fresh_intent=f['id'],fresh_ms=f['t'],fresh_score=f['score'],delay_sec=round((f['t']-e['t'])/1000),
    proxy_closed_bar_newer=True,
    **{f'pending_short_{m}m_pct':e.get(f'short_return_{m}m_pct') for m in (15,30,60)},
    **{f'fresh_short_{m}m_pct':f.get(f'short_return_{m}m_pct') for m in (15,30,60)}))
p=out/'SHORT1_INJ_PENDING_TO_FRESH_READY_5M_PROXY_20261113.csv'
with p.open('w',newline='') as f:
 cols=['pending_intent','pending_ms','pending_score','fresh_intent','fresh_ms','fresh_score','delay_sec','proxy_closed_bar_newer']+[f'{phase}_short_{m}m_pct' for phase in ('pending','fresh') for m in (15,30,60)]
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(matched)
from statistics import mean
print('5M_CANDLE_COUNT',len(prices),'EXEC_INTENT_COUNT',len(events),'PENDING_COUNT',sum(x['gate']=='PENDING_CONTEXT' for x in events),
      'READY_COUNT',sum(x['gate']=='READY' for x in events),'PROXY_MATCHED_FRESH_READY',len(matched))
for horizon in (15,30,60):
 orig=[x[f'pending_short_{horizon}m_pct'] for x in matched if x[f'pending_short_{horizon}m_pct'] is not None]
 fresh=[x[f'fresh_short_{horizon}m_pct'] for x in matched if x[f'fresh_short_{horizon}m_pct'] is not None]
 print('PROXY_MARKOUT_MIN',horizon,'PENDING_MEAN_PCT',round(mean(orig),4) if orig else None,
       'FRESH_MEAN_PCT',round(mean(fresh),4) if fresh else None,'FRESH_WINS',sum(v>0 for v in fresh),'N',len(fresh))
print('RESULT',p)
print('LIMITATION: 5m candle-close proxy, NOT actual fill, fees, Sniper, independent source event ID or executable PENDING promotion')
