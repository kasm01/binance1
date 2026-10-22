#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path('/home/kasm920/binance1'); PRO=Path('/home/kasm920/binance1-pro'); sys.path.insert(0,str(ROOT))
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1,build_short_economic_features
SYMS=['SOLUSDT','DOGEUSDT']; H=20; FAV=.006; ADV=.0075; FRACS=[.02,.05,.10,.15,.20]
COMMON_START=pd.Timestamp('2025-09-19 20:24:00',tz='UTC'); COMMON_END=pd.Timestamp('2026-09-14 19:20:00',tz='UTC'); FOLDS=[(180,30,30),(210,30,30),(240,30,30),(270,30,None)]
START=1000.; LEV=3.; FEE=.0004; TP_ARM=.04; GIVEBACK=.0024; SL=.04; CATA=.06; EXH=-.20; PAUSE_MIN=90
OUT=ROOT/'research/short/results/SHORT2_LIFECYCLE_WF_V1_20261022'; OUT.mkdir(parents=True,exist_ok=True)

def model(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])
def load(sym):
 d3=pd.read_csv(PRO/f'data/offline_cache/{sym}_3m_360d.csv'); d1=pd.read_csv(PRO/f'data/offline_cache/{sym}_1m_360d.csv')
 for c in ['open','high','low','close','open_time','close_time']: d3[c]=pd.to_numeric(d3[c],errors='coerce')
 for c in ['open','high','low','close','quote_asset_volume','taker_buy_quote_volume','open_time']: d1[c]=pd.to_numeric(d1[c],errors='coerce')
 d1['ts']=pd.to_datetime(d1.open_time,unit='ms',utc=True); sell=d1.quote_asset_volume-d1.taker_buy_quote_volume; net=d1.taker_buy_quote_volume-sell; d1['flow60']=net.rolling(60,min_periods=60).sum()/d1.quote_asset_volume.rolling(60,min_periods=60).sum()
 f=build_short_economic_features(d3); o=d3.open.to_numpy(float); hi=d3.high.to_numpy(float); lo=d3.low.to_numpy(float); ct=d3.close_time.to_numpy(np.int64)
 rows=np.arange(0,len(d3)-H-1,dtype=int); ei=rows+1; idx=ei[:,None]+np.arange(H)[None,:]; ent=o[ei]; mfe=np.max(1-lo[idx]/ent[:,None],axis=1); mae=np.max(hi[idx]/ent[:,None]-1,axis=1); y=((mfe>=FAV)&(mae<=ADV)).astype(int)
 X=f.iloc[rows][SHORT_ECON_MODEL_FEATURES_V1].copy(); good=np.isfinite(X.to_numpy(float)).all(axis=1); rows=rows[good]; X=X.loc[good].reset_index(drop=True).to_numpy(float); y=y[good]; ct=ct[rows]; label_end=d3.close_time.to_numpy(np.int64)[np.minimum(rows+H+1,len(d3)-1)]
 return d1,{'X':X,'y':y,'ct':ct,'label_end':label_end}
def window(d,start,end): return d[(d.ts>=start-pd.Timedelta(minutes=65))&(d.ts<end)].copy().reset_index(drop=True)
def score_events(scores,ct,d,start,end,thr):
 opens=d.open_time.to_numpy(np.int64); a=int(start.timestamp()*1000); b=int(end.timestamp()*1000); out={}
 for t,s in zip(ct,scores):
  if t<a or t>=b or s<thr: continue
  j=int(np.searchsorted(opens,int(t)+1,'left'))
  if j<len(d) and a<=int(d.iloc[j].open_time)<b: out[d.iloc[j].ts]=float(s)
 return out

def run(d,score_map):
 eq=START; peak=START; pos=None; candidate=None; last_entry=pd.Timestamp('1970-01-01',tz='UTC'); reentry=last_entry; pause=last_entry; stops=[]; armed_cluster=True; closed=[]; curve=[]
 soft_pending=False
 for i in range(1,len(d)-1):
  r=d.iloc[i]; ts=r.ts; op=float(r.open); hi=float(r.high); lo=float(r.low); cl=float(r.close)
  if pos is not None:
   ent=pos['entry']; qty=pos['qty']; exit_px=None; reason=None
   if hi>=ent*(1+CATA): exit_px=ent*(1+CATA); reason='catastrophic'
   else:
    breach=hi>=ent*(1+SL)
    if breach and soft_pending: exit_px=ent*(1+SL); reason='soft_stop_1m_confirm'
    soft_pending=bool(breach)
    if not breach: soft_pending=False
    if exit_px is None and pos['armed']:
     trig=pos['best_roi']-GIVEBACK; px=ent*(1-trig/LEV)
     if hi>=px: exit_px=px; reason='tp_giveback'
    if exit_px is None:
     roi=((ent-lo)/ent)*LEV
     if roi>pos['best_roi']: pos['best_roi']=roi
     if pos['best_roi']>=TP_ARM: pos['armed']=True
   if exit_px is not None:
    gross=(ent-exit_px)*qty; fee=exit_px*qty*FEE; pnl=gross-fee-pos['open_fee']; eq+=gross-fee; closed.append({'open_ts':pos['open_ts'],'close_ts':ts,'pnl_net':pnl,'reason':reason,'notional':pos['notional'],'score':pos['score']}); peak=max(peak,eq)
    if reason in ('catastrophic','soft_stop_1m_confirm'): stops.append(ts); reentry=max(reentry,ts+pd.Timedelta(minutes=15))
    pos=None; soft_pending=False
  stops=[x for x in stops if x>=ts-pd.Timedelta(hours=6)]
  if len(stops)<3: armed_cluster=True
  if len(stops)>=3 and armed_cluster: pause=max(pause,ts+pd.Timedelta(minutes=PAUSE_MIN)); armed_cluster=False
  if ts in score_map: candidate={'score':score_map[ts],'expires':ts+pd.Timedelta(minutes=1)}
  if candidate is not None and ts<=candidate['expires']:
   prev=d.iloc[i-1]; confirmed=float(prev.close)>float(prev.open) and cl<op and ((op-cl)/op)>=.0001
   if confirmed:
    can=pos is None and ts>=reentry and ts>=pause and (ts-last_entry)>=pd.Timedelta(minutes=3)
    if can and np.isfinite(r.flow60) and float(r.flow60)<=EXH: can=False
    if can:
     er=d.iloc[i+1]; px=float(er.open); notion=eq; fee=notion*FEE; qty=notion/px; eq-=fee; pos={'entry':px,'qty':qty,'notional':notion,'open_fee':fee,'open_ts':er.ts,'score':candidate['score'],'best_roi':0.,'armed':False}; last_entry=ts
    candidate=None
  elif candidate is not None and ts>candidate['expires']: candidate=None
  mtm=eq if pos is None else eq+(pos['entry']-cl)*pos['qty']; curve.append((ts,eq,mtm))
 C=pd.DataFrame(closed); Q=pd.DataFrame(curve,columns=['ts','equity','mtm']); p=C.pnl_net.to_numpy(float) if len(C) else np.array([]); gp=p[p>0].sum(); gl=-p[p<0].sum(); arr=Q.mtm.to_numpy(float); pk=np.maximum.accumulate(arr); dd=np.min(arr/pk-1)*100
 return {'return_pct':(eq/START-1)*100,'pf':gp/max(gl,1e-12),'dd_pct':dd,'closed':len(C),'win_pct':100*(p>0).mean() if len(p) else 0.,'fees_proxy':float(sum(x['notional']*FEE*2 for x in closed))},C,Q

def main():
 rows=[]
 for sym in SYMS:
  print('\nSYMBOL',sym,flush=True); d1,d=load(sym)
  for fold,(td,vd,xd) in enumerate(FOLDS,1):
   te=COMMON_START+pd.Timedelta(days=td); ve=te+pd.Timedelta(days=vd); xe=COMMON_END if xd is None else min(ve+pd.Timedelta(days=xd),COMMON_END); tr=(d['ct']>=int(COMMON_START.timestamp()*1000))&(d['label_end']<int(te.timestamp()*1000)); m=model().fit(d['X'][tr],d['y'][tr]); pall=m.predict_proba(d['X'])[:,1]; ptr=pall[tr]
   dv=window(d1,te,ve); choices=[]
   for frac in FRACS:
    th=float(np.quantile(ptr,1-frac)); sm=score_events(pall,d['ct'],dv,te,ve,th); s,_,_=run(dv,sm)
    if s['closed']<3: continue
    obj=s['return_pct']+.15*s['dd_pct']; choices.append((obj,s['return_pct'],s['dd_pct'],s['closed'],frac,th))
   if not choices: print('FOLD',fold,'NO_VALID',flush=True); continue
   choices.sort(reverse=True); obj,vr,vdd,vn,frac,th=choices[0]; dt=window(d1,ve,xe); sm=score_events(pall,d['ct'],dt,ve,xe,th); s,C,Q=run(dt,sm); row={'symbol':sym,'fold':fold,'top_fraction':frac,'threshold':th,'val_obj':obj,'val_return':vr,'val_dd':vdd,'val_closed':vn,'test_return':s['return_pct'],'test_pf':s['pf'],'test_dd':s['dd_pct'],'test_closed':s['closed'],'test_win':s['win_pct']}; rows.append(row); print('FOLD_RESULT',row,flush=True)
 R=pd.DataFrame(rows); R.to_csv(OUT/'folds.csv',index=False)
 if len(R):
  A=R.groupby('symbol').agg(folds=('fold','count'),positive_folds=('test_return',lambda x:int((x>0).sum())),return_mean=('test_return','mean'),return_median=('test_return','median'),worst_fold=('test_return','min'),pf_mean=('test_pf','mean'),dd_mean=('test_dd','mean'),dd_worst=('test_dd','min'),closed_mean=('test_closed','mean')).reset_index(); A.to_csv(OUT/'summary.csv',index=False); print('\nSUMMARY\n'+A.to_string(index=False),flush=True)
if __name__=='__main__':main()
