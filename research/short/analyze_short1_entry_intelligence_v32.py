#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

ROOT=Path('/home/kasm920/binance1')
ONE=ROOT/'research/short/data/inj_1m'
OI=ROOT/'research/short/data/INJUSDT_metrics_5m_20251012_20261013.csv'
BASE=ROOT/'research/short/results/SHORT1_COMBINED_V30/C600'
OUT=ROOT/'research/short/results/SHORT1_ENTRY_INTELLIGENCE_V32'
OUT.mkdir(parents=True,exist_ok=True)


def load_1m():
    fs=sorted(ONE.glob('*.csv'))
    d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True)
    nums=['open','high','low','close','quote_volume','taker_buy_quote_volume']
    for c in nums: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    q=d.quote_volume.replace(0,np.nan)
    buy=d.taker_buy_quote_volume
    net=2*buy-d.quote_volume
    d['flow1']=net/q
    d['ret1']=d.close.pct_change(1)
    for w in [3,5,15,60]:
        d[f'flow{w}']=net.rolling(w,min_periods=w).sum()/d.quote_volume.rolling(w,min_periods=w).sum()
        d[f'ret{w}']=d.close.pct_change(w)
    d['buy_ratio']=buy/q
    d['body_pct']=(d.close-d.open)/d.open
    rng=(d.high-d.low).replace(0,np.nan)
    d['range_pct']=rng/d.open
    d['close_loc']=(d.close-d.low)/rng
    d['upper_wick_pct']=(d.high-np.maximum(d.open,d.close))/d.open
    d['lower_wick_pct']=(np.minimum(d.open,d.close)-d.low)/d.open
    vm=d.quote_volume.rolling(120,min_periods=30).mean().shift(1)
    vs=d.quote_volume.rolling(120,min_periods=30).std(ddof=0).shift(1).replace(0,np.nan)
    d['volz120']=(d.quote_volume-vm)/vs
    # OI causal snapshots
    m=pd.read_csv(OI)
    m['oi_ts']=pd.to_datetime(m.create_time,utc=True)
    m=m.sort_values('oi_ts').drop_duplicates('oi_ts').reset_index(drop=True)
    soi=pd.to_numeric(m.sum_open_interest,errors='coerce')
    m['oi5']=soi.pct_change(1)
    m['oi15']=soi.pct_change(3)
    d['lookup']=d.ts-pd.Timedelta(nanoseconds=1)
    d=pd.merge_asof(d.sort_values('lookup'),m[['oi_ts','oi5','oi15']],left_on='lookup',right_on='oi_ts',direction='backward',tolerance=pd.Timedelta('6m'))
    return d.sort_values('ts').reset_index(drop=True)


def first_touch(win, entry, fav_thr, adv_thr):
    # SHORT: favorable is low <= entry*(1-fav), adverse high >= entry*(1+adv)
    fav_px=entry*(1-fav_thr); adv_px=entry*(1+adv_thr)
    for _,r in win.iterrows():
        f=float(r.low)<=fav_px; a=float(r.high)>=adv_px
        if f and a: return 'BOTH_SAME_BAR'
        if f: return 'FAVORABLE_FIRST'
        if a: return 'ADVERSE_FIRST'
    return 'NEITHER'


def main():
    d=load_1m(); idx={t:i for i,t in enumerate(d.ts)}
    E=pd.read_csv(BASE/'entries.csv'); E['ts']=pd.to_datetime(E.ts,utc=True)
    C=pd.read_csv(BASE/'closed_positions.csv'); C['open_ts']=pd.to_datetime(C.open_ts,utc=True); C['close_ts']=pd.to_datetime(C.close_ts,utc=True)
    # one-position architecture makes open_ts unique enough; keep first if duplicate
    cm=C.sort_values('close_ts').drop_duplicates('open_ts').set_index('open_ts')
    rows=[]
    for _,e in E.iterrows():
        t=e.ts
        if t not in idx: continue
        i=idx[t]
        if i<20 or i+5>=len(d): continue
        entry=float(e.price)
        pre=d.iloc[i-1]  # last fully completed 1m bar before execution at current minute open
        r={
            'entry_ts':t,'entry_price':entry,'score':float(e.score),'notional':float(e.notional),
            'flow1':float(pre.flow1),'flow3':float(pre.flow3),'flow5':float(pre.flow5),'flow15':float(pre.flow15),'flow60':float(pre.flow60),
            'flow_accel_1_5':float(pre.flow1-pre.flow5),'flow_accel_3_15':float(pre.flow3-pre.flow15),
            'ret1':float(pre.ret1),'ret3':float(pre.ret3),'ret5':float(pre.ret5),'ret15':float(pre.ret15),
            'buy_ratio':float(pre.buy_ratio),'body_pct':float(pre.body_pct),'range_pct':float(pre.range_pct),'close_loc':float(pre.close_loc),
            'upper_wick_pct':float(pre.upper_wick_pct),'lower_wick_pct':float(pre.lower_wick_pct),'volz120':float(pre.volz120),
            'oi5':float(pre.oi5) if np.isfinite(pre.oi5) else np.nan,'oi15':float(pre.oi15) if np.isfinite(pre.oi15) else np.nan,
        }
        for w in [1,3,5]:
            # price response efficiency: directional price response per absolute flow pressure over prior completed window
            flow=r[f'flow{w}'] if w in [1,3,5] else np.nan
            ret=r[f'ret{w}']
            r[f'flow_eff_{w}']=ret/(abs(flow)+1e-4)
            fw=d.iloc[i:i+w]
            r[f'mae_{w}m_pct']=(float(fw.high.max())/entry-1)*100
            r[f'mfe_{w}m_pct']=(1-float(fw.low.min())/entry)*100
            r[f'close_ret_{w}m_pct']=(float(fw.close.iloc[-1])/entry-1)*100
        fw5=d.iloc[i:i+5]
        for th in [0.0025,0.0040,0.0060]:
            r[f'touch_{int(th*10000)}bp_5m']=first_touch(fw5,entry,th,th)
        if t in cm.index:
            c=cm.loc[t]
            r['pnl_net']=float(c.pnl_net); r['trade_win']=int(float(c.pnl_net)>0); r['exit_reason']=str(c.reason)
        else:
            r['pnl_net']=np.nan; r['trade_win']=np.nan; r['exit_reason']='OPEN_OR_UNMATCHED'
        # early bad-entry labels independent of eventual lifecycle outcome
        r['bad_entry_3m']=int(r['mae_3m_pct']>=0.40 and r['mfe_3m_pct']<0.25)
        r['bad_entry_5m']=int(r['mae_5m_pct']>=0.40 and r['mfe_5m_pct']<0.40)
        r['good_entry_3m']=int(r['mfe_3m_pct']>=0.40 and r['mae_3m_pct']<0.25)
        rows.append(r)
    X=pd.DataFrame(rows)
    X.to_csv(OUT/'entry_intelligence_dataset.csv',index=False)

    feats=['score','flow1','flow3','flow5','flow15','flow60','flow_accel_1_5','flow_accel_3_15','ret1','ret3','ret5','ret15','buy_ratio','body_pct','range_pct','close_loc','upper_wick_pct','lower_wick_pct','volz120','oi5','oi15','flow_eff_1','flow_eff_3','flow_eff_5']
    summary={'rows':len(X),'matched_closed':int(X.trade_win.notna().sum()),'wins':int((X.trade_win==1).sum()),'losses':int((X.trade_win==0).sum()),
             'bad_entry_3m':int(X.bad_entry_3m.sum()),'bad_entry_5m':int(X.bad_entry_5m.sum()),'good_entry_3m':int(X.good_entry_3m.sum())}
    print('SUMMARY',json.dumps(summary,sort_keys=True))
    print('\nEARLY EXCURSION MEDIANS BY FINAL OUTCOME')
    print(X.groupby('trade_win')[['mae_1m_pct','mfe_1m_pct','mae_3m_pct','mfe_3m_pct','mae_5m_pct','mfe_5m_pct']].median().round(4).to_string())
    print('\nFIRST TOUCH 40BP / 5M')
    print(pd.crosstab(X['touch_40bp_5m'],X.trade_win,normalize='columns').round(4).to_string())
    print('\nFEATURE EFFECTS: BAD_ENTRY_3M vs OTHERS')
    out=[]
    for f in feats:
        a=pd.to_numeric(X.loc[X.bad_entry_3m==1,f],errors='coerce').dropna(); b=pd.to_numeric(X.loc[X.bad_entry_3m==0,f],errors='coerce').dropna()
        if len(a)<10 or len(b)<10: continue
        sd=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        eff=(a.mean()-b.mean())/sd if sd>0 else np.nan
        out.append((f,len(a),a.mean(),b.mean(),eff))
    T=pd.DataFrame(out,columns=['feature','bad_n','bad_mean','other_mean','std_effect']).sort_values('std_effect',key=lambda s:s.abs(),ascending=False)
    T.to_csv(OUT/'feature_effects_bad_entry_3m.csv',index=False)
    print(T.head(20).round(6).to_string(index=False))

    print('\nFEATURE EFFECTS: FINAL LOSERS vs WINNERS')
    out=[]
    Z=X[X.trade_win.notna()].copy()
    for f in feats:
        a=pd.to_numeric(Z.loc[Z.trade_win==0,f],errors='coerce').dropna(); b=pd.to_numeric(Z.loc[Z.trade_win==1,f],errors='coerce').dropna()
        if len(a)<10 or len(b)<10: continue
        sd=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        eff=(a.mean()-b.mean())/sd if sd>0 else np.nan
        out.append((f,len(a),a.mean(),b.mean(),eff))
    T2=pd.DataFrame(out,columns=['feature','loss_n','loss_mean','win_mean','std_effect']).sort_values('std_effect',key=lambda s:s.abs(),ascending=False)
    T2.to_csv(OUT/'feature_effects_final_loss.csv',index=False)
    print(T2.head(20).round(6).to_string(index=False))

if __name__=='__main__': main()
