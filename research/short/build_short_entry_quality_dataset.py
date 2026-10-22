#!/usr/bin/env python3
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from features.short_economic import build_short_economic_features, SHORT_ECON_MODEL_FEATURES_V1
SRC3=Path('/home/kasm920/binance1-pro/data/offline_cache')
SRC1=ROOT/'research/short/data/real_confirm_window_1m'
EVENTS=ROOT/'research/short/results/SHORT_REAL_ENTRY_STATE_EVENTS_15S.csv'
OUT=ROOT/'research/short/results/SHORT_ENTRY_QUALITY_FEATURES.csv'
OBS_MIN=60; FAV_PCT=.0060; ADV_MAX_PCT=.0075

def main():
    e=pd.read_csv(EVENTS); e['ts']=pd.to_datetime(e.ts,utc=True)
    rows=[]
    for sym,g in e.groupby('symbol'):
        p1=SRC1/f'{sym}_1m.csv'; p3=SRC3/f'{sym}_3m_360d.csv'
        if not p1.exists() or not p3.exists(): continue
        d1=pd.read_csv(p1); ot1=pd.to_numeric(d1.open_time).to_numpy('int64')
        hi1=pd.to_numeric(d1.high).to_numpy(float); lo1=pd.to_numeric(d1.low).to_numpy(float); cl1=pd.to_numeric(d1.close).to_numpy(float)
        d3=pd.read_csv(p3); f3=build_short_economic_features(d3); ct3=pd.to_numeric(d3.close_time).to_numpy('int64')
        for _,r in g.iterrows():
            tms=int(r.ts.timestamp()*1000); minute=(tms//60000)*60000
            si=int(np.searchsorted(ot1,minute+60000,side='left')); end=min(len(d1),si+OBS_MIN)
            if si>=len(d1) or end-si<OBS_MIN: continue
            fi=int(np.searchsorted(ct3,tms,side='left')-1)
            if fi<0: continue
            entry=float(r.entry_price); hh=hi1[si:end]; ll=lo1[si:end]
            mfe=float(np.max(1.0-ll/entry)); mae=float(np.max(hh/entry-1.0)); end_ret=float(1.0-cl1[end-1]/entry)
            target=int(mfe>=FAV_PCT and mae<=ADV_MAX_PCT)
            z=r.to_dict(); z.update(observation_min=OBS_MIN,mfe_pct=mfe,mae_pct=mae,end_return_pct=end_ret,entry_quality_target=target,feature_bar_index=fi,feature_close_time=int(ct3[fi]))
            for c in SHORT_ECON_MODEL_FEATURES_V1: z[c]=f3.iloc[fi][c]
            rows.append(z)
    z=pd.DataFrame(rows).sort_values('ts').reset_index(drop=True); z.to_csv(OUT,index=False)
    print('ROWS',len(z),'FEATURES',len(SHORT_ECON_MODEL_FEATURES_V1),'TARGET_RATE',round(z.entry_quality_target.mean(),4))
    for state,g in z.groupby('state'):
        print(state,'n',len(g),'target_rate',round(g.entry_quality_target.mean(),4),'mfe_bps',round(g.mfe_pct.mean()*1e4,2),'mae_bps',round(g.mae_pct.mean()*1e4,2))
if __name__=='__main__': main()
