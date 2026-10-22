#!/usr/bin/env python3
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path('/home/kasm920/binance1'); PRO=Path('/home/kasm920/binance1-pro'); sys.path.insert(0,str(ROOT))
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, build_short_economic_features
import research.short.walkforward_short2_lifecycle_v1 as wf
SYM='DOGEUSDT'; H=20; FAV=.006; ADV=.0075
TRAIN_START=pd.Timestamp('2025-09-19 20:24:00',tz='UTC'); TRAIN_END=pd.Timestamp('2026-08-15 19:20:00',tz='UTC'); VAL_END=pd.Timestamp('2026-09-14 19:20:00',tz='UTC'); TEST_END=pd.Timestamp('2026-10-11 21:36:00',tz='UTC')
FRACS=[.02,.05,.10,.15,.20]; OUT=ROOT/'research/short/results/SHORT2_DOGE_PROMOTION_FORWARD_V1_20261022'; OUT.mkdir(parents=True,exist_ok=True)
def model(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])
def union1():
 d=pd.concat([pd.read_csv(PRO/f'data/offline_cache/{SYM}_1m_360d.csv'),pd.read_csv(PRO/f'research/data/{SYM}_1m_1month.csv')],ignore_index=True).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
 for c in ['open','high','low','close','quote_asset_volume','taker_buy_quote_volume','open_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
 d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True); sell=d.quote_asset_volume-d.taker_buy_quote_volume; net=d.taker_buy_quote_volume-sell; d['flow60']=net.rolling(60,min_periods=60).sum()/d.quote_asset_volume.rolling(60,min_periods=60).sum(); return d
def union3():
 h=pd.read_csv(PRO/f'data/offline_cache/{SYM}_3m_360d.csv'); m=pd.read_csv(PRO/f'research/data/{SYM}_1m_1month.csv')
 for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','open_time','close_time']: m[c]=pd.to_numeric(m[c],errors='coerce')
 m['bucket']=pd.to_datetime(m.open_time,unit='ms',utc=True).dt.floor('3min'); a=m.groupby('bucket').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),open_time=('open_time','first'),close_time=('close_time','last'),quote_asset_volume=('quote_asset_volume','sum'),number_of_trades=('number_of_trades','sum'),taker_buy_base_volume=('taker_buy_base_volume','sum'),taker_buy_quote_volume=('taker_buy_quote_volume','sum'),n=('open_time','size')).reset_index(drop=True); a=a[a.n==3].drop(columns='n'); a['ignore']=0
 d=pd.concat([h,a],ignore_index=True).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True); return d
def prep3():
 d=union3();
 for c in ['open','high','low','close','open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
 f=build_short_economic_features(d); o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float); ct=d.close_time.to_numpy(np.int64)
 rows=np.arange(0,len(d)-H-1,dtype=int); ei=rows+1; idx=ei[:,None]+np.arange(H)[None,:]; ent=o[ei]; mfe=np.max(1-lo[idx]/ent[:,None],axis=1); mae=np.max(hi[idx]/ent[:,None]-1,axis=1); y=((mfe>=FAV)&(mae<=ADV)).astype(int)
 X=f.iloc[rows][SHORT_ECON_MODEL_FEATURES_V1].copy(); good=np.isfinite(X.to_numpy(float)).all(axis=1); rows=rows[good]; return X.loc[good].reset_index(drop=True).to_numpy(float),y[good],ct[rows],d.close_time.to_numpy(np.int64)[np.minimum(rows+H+1,len(d)-1)]
def main():
 d1=union1(); X,y,ct,label_end=prep3(); tr=(ct>=int(TRAIN_START.timestamp()*1000))&(label_end<int(TRAIN_END.timestamp()*1000)); m=model().fit(X[tr],y[tr]); p=m.predict_proba(X)[:,1]; ptr=p[tr]
 dv=wf.window(d1,TRAIN_END,VAL_END); rows=[]
 for frac in FRACS:
  th=float(np.quantile(ptr,1-frac)); ev=wf.score_events(p,ct,dv,TRAIN_END,VAL_END,th); s,_,_=wf.run(dv,ev); obj=s['return_pct']+.15*s['dd_pct']; rows.append({'top_fraction':frac,'threshold':th,'objective':obj,**s})
 V=pd.DataFrame(rows); V.to_csv(OUT/'validation_grid.csv',index=False); print('VALIDATION\n'+V.to_string(index=False),flush=True)
 E=V[(V.closed>=5)&(V.return_pct>0)&(V.dd_pct>-10)&(V.pf>1)].copy()
 if E.empty: (OUT/'decision.txt').write_text('NO_DEPLOY\n'); print('PROMOTION NO_DEPLOY reason=no_validation_profile_passed',flush=True); return
 best=E.sort_values(['objective','return_pct','pf'],ascending=False).iloc[0]; frac=float(best.top_fraction); th=float(best.threshold); print('PROMOTION PASS',best.to_dict(),flush=True)
 dt=wf.window(d1,VAL_END,TEST_END); ev=wf.score_events(p,ct,dt,VAL_END,TEST_END,th); s,C,Q=wf.run(dt,ev); print('FORWARD',s,flush=True); C.to_csv(OUT/'forward_closed.csv',index=False); Q.to_csv(OUT/'forward_curve.csv',index=False)
 decision='SHADOW_CANDIDATE' if (s['return_pct']>0 and s['pf']>1 and s['dd_pct']>-10 and s['closed']>=5) else 'NO_DEPLOY_FORWARD_FAIL'; (OUT/'decision.txt').write_text(f'{decision}\nfrac={frac}\nthreshold={th}\nvalidation={best.to_dict()}\nforward={s}\n'); print('FINAL_DECISION',decision,flush=True)
if __name__=='__main__': main()
