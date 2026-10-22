#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT_PATH=Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path: sys.path.insert(0,str(ROOT_PATH))
import json, os, time
import numpy as np, pandas as pd, joblib, redis, requests
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, build_short_economic_features
ROOT=Path('/home/kasm920/binance1'); OFF=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv'); MODEL=ROOT/'research/short/models/short_model_INJUSDT_3m_fresh_v5.joblib'; OUT=ROOT/'research/short/results/SHORT1_INJ_FRESH_V5_SHADOW_SCORES.csv'; STREAM='telemetry:short1:inj:model_shadow_v3'
r=redis.Redis(host='localhost',port=6379,db=0,decode_responses=True); events=[]
for sid,f in reversed(r.xrevrange(STREAM,count=20000)):
 try:
  o=json.loads(f.get('json','{}')); events.append({'event_id':o.get('event_id'),'source_ts':float(o.get('ts') or 0),'old_score':float(o.get('economic_probability') or 0),'confirmation_state':o.get('confirmation_state')})
 except Exception: pass
ev=pd.DataFrame(events).drop_duplicates('event_id').sort_values('source_ts').reset_index(drop=True); assert len(ev)
off=pd.read_csv(OFF); last=int(pd.to_numeric(off.open_time).max()); start=last+180000; end=int(time.time()*1000); cols=list(off.columns); ses=requests.Session(); rows=[]; cur=start
while cur<=end:
 rr=ses.get('https://fapi.binance.com/fapi/v1/klines',params={'symbol':'INJUSDT','interval':'3m','startTime':cur,'endTime':end,'limit':1500},timeout=10); rr.raise_for_status(); b=rr.json()
 if not b: break
 rows.extend(b); cur=int(b[-1][0])+180000
 if len(b)<1500: break
 time.sleep(.05)
# Normalize fetched Binance columns to offline schema names.
base_cols=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
fw=pd.DataFrame(rows,columns=base_cols)
# offline feature builder needs same core fields; concatenate only common columns.
need=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
for c in need:
 if c not in off.columns and c=='quote_asset_volume' and 'quote_volume' in off.columns: off[c]=off['quote_volume']
all3=pd.concat([off[need].tail(600),fw[need]],ignore_index=True).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
for c in ['open','high','low','close','open_time','close_time']: all3[c]=pd.to_numeric(all3[c],errors='coerce')
feat=build_short_economic_features(all3); m=joblib.load(MODEL); scores=[]
close_ms=pd.to_numeric(all3.close_time).to_numpy(np.int64)
for _,e in ev.iterrows():
 tms=int(e.source_ts*1000); idx=int(np.searchsorted(close_ms,tms,side='left')-1)
 if idx<0: scores.append(np.nan); continue
 x=feat.iloc[[idx]][SHORT_ECON_MODEL_FEATURES_V1]
 if not np.isfinite(x.to_numpy(float)).all(): scores.append(np.nan); continue
 scores.append(float(m.predict_proba(x)[:,1][0]))
ev['fresh_score']=scores; ev.to_csv(OUT,index=False)
z=ev.dropna(subset=['fresh_score']); print('EVENTS',len(ev),'SCORED',len(z),'OLD_MEAN',z.old_score.mean(),'FRESH_MEAN',z.fresh_score.mean(),'CORR',z[['old_score','fresh_score']].corr().iloc[0,1])
for name,col in [('OLD','old_score'),('FRESH','fresh_score')]:
 c=z[z.confirmation_state=='CONFIRM'][col]; print(name,'CONFIRM_N',len(c),'mean',c.mean(),'hard>=.54',int((c>=.54).sum()),'soft .53-.54',int(((c>=.53)&(c<.54)).sum()))
print('OUT',OUT)
