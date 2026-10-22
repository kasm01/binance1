#!/usr/bin/env python3
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path('/home/kasm920/binance1'); sys.path.insert(0,str(ROOT))
from research.short.short_economic_features import build_short_economic_features, SHORT_ECON_FEATURES
SRC3=Path('/home/kasm920/binance1-pro/data/offline_cache')
SRC1=ROOT/'research/short/data/real_confirm_window_1m'
EVENTS=ROOT/'research/short/results/SHORT_REAL_ENTRY_STATE_EVENTS_15S.csv'
OUT=ROOT/'research/short/results/SHORT_CANDIDATE_LIFECYCLE_FEATURES.csv'
LEV=3.0; TP_ROI=.0120; GIVEBACK=.0024; SOFT_ADVERSE_PCT=.030; CATASTROPHIC_PCT=.045; FEE=.0008; LABEL_OBSERVATION_MIN=180

def simulate(d, start_i, entry):
    best=0.0; armed=False; soft_pending=False
    soft_px=entry*(1.0+SOFT_ADVERSE_PCT)
    cat_px=entry*(1.0+CATASTROPHIC_PCT)
    end_i=min(len(d), start_i + LABEL_OBSERVATION_MIN)
    for k in range(start_i,end_i):
        hi=float(d.iloc[k].high); lo=float(d.iloc[k].low); cl=float(d.iloc[k].close)
        # Catastrophic protection is immediate.
        if hi>=cat_px:
            gross=(entry-cat_px)/entry
            return gross-FEE,'dynamic_stop_catastrophic',k-start_i+1,best
        # 60s soft-stop confirmation proxy on 1m bars:
        # crossing arms pending; close must still be adverse at bar close.
        if hi>=soft_px:
            soft_pending=True
        if soft_pending:
            if cl>=soft_px:
                gross=(entry-cl)/entry
                return gross-FEE,'dynamic_stop_confirmed',k-start_i+1,best
            soft_pending=False
        # Existing armed giveback is checked before a new favorable low.
        if armed:
            trigger=best-GIVEBACK
            trig_px=entry*(1.0-trigger/LEV)
            if hi>=trig_px:
                gross=(entry-trig_px)/entry
                return gross-FEE,'tp_giveback',k-start_i+1,best
        low_roi=((entry-lo)/entry)*LEV
        if low_roi>best: best=low_roi
        if best>=TP_ROI: armed=True
    # No production time exit. End-of-data rows are censored, not outcomes.
    return float('nan'),'censored',end_i-start_i,best

def main():
    e=pd.read_csv(EVENTS); e['ts']=pd.to_datetime(e.ts,utc=True)
    rows=[]
    for sym,g in e.groupby('symbol'):
        d1=pd.read_csv(SRC1/f'{sym}_1m.csv'); ot1=pd.to_numeric(d1.open_time).to_numpy('int64')
        d3=pd.read_csv(SRC3/f'{sym}_3m_360d.csv'); f3=build_short_economic_features(d3); ct3=pd.to_numeric(d3.close_time).to_numpy('int64')
        for _,r in g.iterrows():
            tms=int(r.ts.timestamp()*1000); minute=(tms//60000)*60000
            # outcome begins at next FULL minute to avoid using pre-snapshot path of current minute
            si=int(np.searchsorted(ot1,minute+60000,side='left'))
            if si>=len(d1): continue
            fi=int(np.searchsorted(ct3,tms,side='left')-1)
            if fi<0: continue
            net,outcome,mins,best=simulate(d1,si,float(r.entry_price))
            z=r.to_dict(); z.update(net_return_pct=net,outcome=outcome,exit_minutes=mins,best_roi=best,feature_bar_index=fi,feature_close_time=int(ct3[fi]))
            for c in SHORT_ECON_FEATURES: z[c]=f3.iloc[fi][c]
            rows.append(z)
    z=pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
    z.to_csv(OUT,index=False)
    print('ROWS',len(z),'FEATURES',len(SHORT_ECON_FEATURES),'NAN',int(z[SHORT_ECON_FEATURES].isna().any(axis=1).sum()))
    print(z.groupby(['state','outcome']).size().to_string())
    for state,g in list(z.groupby('state'))+[('ALL',z)]:
        a=g.net_return_pct.to_numpy(float); pos=a[a>0].sum(); neg=-a[a<0].sum()
        print(state,'n',len(g),'net_bps',round(a.mean()*1e4,3),'win',round((a>0).mean(),3),'pf',round(pos/max(neg,1e-12),3))
if __name__=='__main__': main()
