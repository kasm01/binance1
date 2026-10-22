#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

EVENTS=Path('research/short/results/SHORT_REAL_CONFIRM_EVENTS.csv')
DATADIR=Path('research/short/data/real_confirm_window_1m')
OUT=Path('research/short/results')
FEE=0.0008
CONFIGS=[
 ('current18',0.00275,0.003,18),
 ('wide60',0.006,0.010,60),
 ('wide120',0.006,0.010,120),
 ('runner120',0.012,0.010,120),
]

def one_event(d, ev, tp, sl, H):
    ts=pd.Timestamp(ev.ts)
    if ts.tzinfo is None: ts=ts.tz_localize('UTC')
    minute_ms=int(ts.floor('min').timestamp()*1000)
    # start only at the NEXT full 1m candle; avoids pre-confirm high/low contamination
    start_ms=minute_ms+60_000
    pos=d.index[d.open_time==start_ms]
    if len(pos)!=1: return None
    i=int(pos[0]); w=d.iloc[i:i+H]
    if len(w)<H: return None
    e=float(ev.entry_price)
    hi=w.high.to_numpy(float); lo=w.low.to_numpy(float); cl=w.close.to_numpy(float)
    slh=(hi/e-1)>=sl; tph=(1-lo/e)>=tp
    si=int(np.argmax(slh)) if slh.any() else H+1
    ti=int(np.argmax(tph)) if tph.any() else H+1
    # conservative same-minute tie -> stop
    if si<=ti and si<H:
        outcome='sl'; hit_bar=si+1; gross=-sl
    elif ti<si and ti<H:
        outcome='tp'; hit_bar=ti+1; gross=tp
    else:
        outcome='timeout'; hit_bar=H; gross=1-float(cl[-1])/e
    mfe=float(np.max(1-lo/e)); mae=float(np.max(hi/e-1))
    net=gross-FEE
    return dict(outcome=outcome,hit_bar=hit_bar,gross_return_pct=gross,net_return_pct=net,mfe_pct=mfe,mae_pct=mae)

def metrics(g):
    r=g.net_return_pct.to_numpy(float); pos=r[r>0].sum(); neg=-r[r<0].sum()
    return dict(n=len(g),net_bps=float(r.mean()*1e4),win=float((r>0).mean()),pf=float(pos/max(neg,1e-12)),
                tp_rate=float((g.outcome=='tp').mean()),sl_rate=float((g.outcome=='sl').mean()),timeout_rate=float((g.outcome=='timeout').mean()),
                mfe_bps=float(g.mfe_pct.mean()*1e4),mae_bps=float(g.mae_pct.mean()*1e4),mean_hit_bar=float(g.hit_bar.mean()),
                mean_reversal_bps=float(g.reversal_pct.mean()*1e4))

def main():
    E=pd.read_csv(EVENTS); E['ts']=pd.to_datetime(E.ts,utc=True)
    ds={}
    for sym in E.symbol.unique():
        d=pd.read_csv(DATADIR/f'{sym}_1m.csv')
        for c in ['open_time','open','high','low','close']: d[c]=pd.to_numeric(d[c])
        d=d.reset_index(drop=True); ds[sym]=d
    rows=[]
    for _,ev in E.iterrows():
        d=ds[ev.symbol]
        for name,tp,sl,H in CONFIGS:
            z=one_event(d,ev,tp,sl,H)
            if z is not None: rows.append(dict(ts=ev.ts,symbol=ev.symbol,entry_price=ev.entry_price,reversal_pct=ev.reversal_pct,config=name,tp=tp,sl=sl,H=H,**z))
    R=pd.DataFrame(rows); R.to_csv(OUT/'SHORT_REAL_CONFIRM_OUTCOMES.csv',index=False)
    print('EVENTS',len(E),'LABELED_UNIQUE',R[['ts','symbol']].drop_duplicates().shape[0],flush=True)
    summary=[]
    for (cfg,sym),g in R.groupby(['config','symbol']): summary.append(dict(config=cfg,symbol=sym,**metrics(g)))
    for cfg,g in R.groupby('config'): summary.append(dict(config=cfg,symbol='ALL',**metrics(g)))
    S=pd.DataFrame(summary); S.to_csv(OUT/'SHORT_REAL_CONFIRM_OUTCOME_SUMMARY.csv',index=False)
    print('\nSUMMARY'); print(S.to_string(index=False),flush=True)
    # Diagnostic only: does stronger early reversal correlate with better economics?
    base=R[R.config=='current18'].copy()
    base['reversal_bucket']=pd.qcut(base.reversal_pct,4,duplicates='drop')
    print('\nREVERSAL_BUCKETS current18')
    for b,g in base.groupby('reversal_bucket',observed=True): print(str(b),metrics(g),flush=True)

if __name__=='__main__': main()
