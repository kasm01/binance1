#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np, pandas as pd
ROOT=Path('/home/kasm920/binance1')
ONE=ROOT/'research/short/data/inj_1m'
SCORES=ROOT/'research/short/results/SHORT1_INJ_WF_SCORES_V4.csv'
OUT=ROOT/'research/short/results/SHORT1_MIXED_1M_ARCHFIX_V15'; OUT.mkdir(parents=True,exist_ok=True)
START=1000.0; LEV=3.0; FEE_SIDE=0.0004; TP_ARM=.0400; GIVEBACK=.0024; SL_PRICE=.0299; CATA=.035
NORMAL=(1.00,.50,2.00); REDUCED=(.50,.10,.70); DD_RED=.08; DD_REC=.04
BASE_TH=.540; L2_N=.545; L2_C=.550; COOLDOWN_MIN=3; PAUSE_MIN=90

def load1m():
    fs=sorted(ONE.glob('*.csv'))
    parts=[pd.read_csv(f) for f in fs]
    d=pd.concat(parts,ignore_index=True)
    d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True)
    for c in ['open','high','low','close','quote_volume','taker_buy_quote_volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    sell=d.quote_volume-d.taker_buy_quote_volume
    net=d.taker_buy_quote_volume-sell
    d['flow15']=net.rolling(15,min_periods=15).sum()/d.quote_volume.rolling(15,min_periods=15).sum()
    d['flow60']=net.rolling(60,min_periods=60).sum()/d.quote_volume.rolling(60,min_periods=60).sum()
    vm=d.quote_volume.rolling(720,min_periods=240).mean().shift(1); vs=d.quote_volume.rolling(720,min_periods=240).std(ddof=0).shift(1)
    d['volz']=(d.quote_volume-vm)/vs.replace(0,np.nan)
    return d

def load_scores():
    s=pd.read_csv(SCORES); s['score_ts']=pd.to_datetime(s.ts,utc=True); s['eligible_ts']=s.score_ts+pd.Timedelta(minutes=3)
    return s[['score_ts','eligible_ts','wf_score']]

def monthly(curve,closed):
    c=curve.copy(); c['month']=pd.to_datetime(c.ts,utc=True).dt.to_period('M').astype(str)
    e=c.groupby('month').equity.last(); prev=e.shift(1); prev.iloc[0]=START
    rows=[]
    cc=closed.copy(); cc['month']=pd.to_datetime(cc.close_ts,utc=True).dt.to_period('M').astype(str) if len(cc) else []
    for m,v in e.items():
        x=cc[cc.month==m] if len(cc) else cc
        p=x.pnl_net.to_numpy(float) if len(x) else np.array([]); gp=p[p>0].sum() if len(p) else 0; gl=-p[p<0].sum() if len(p) else 0
        rows.append({'month':m,'return_pct':(v/prev.loc[m]-1)*100,'closed':len(x),'pf':gp/gl if gl>0 else np.nan})
    return pd.DataFrame(rows)

def run(d,s,exh=None,name='BASE'):
    score_map={t:float(v) for t,v in zip(s.eligible_ts,s.wf_score) if np.isfinite(v)}
    eq=START; peak=eq; mode='NORMAL'; pos=None; last_entry_ts=pd.Timestamp('1970-01-01',tz='UTC'); pause_until=last_entry_ts
    stop_times=[]; candidate=None; entries=[]; closed=[]; curve=[]; blocked_exh=blocked_l2=blocked_pause=0
    soft_pending=False; equity_hist=[]; stop_cluster_armed=True
    for i in range(1,len(d)-1):
        r=d.iloc[i]; ts=r.ts; op=float(r.open); hi=float(r.high); lo=float(r.low); cl=float(r.close)
        # lifecycle on 1m path
        if pos is not None:
            ent=pos['avg']; qty=pos['qty']; exit_px=None; reason=None
            if hi>=ent*(1+CATA): exit_px=ent*(1+CATA); reason='catastrophic'
            else:
                breach=hi>=ent*(1+SL_PRICE)
                # 30s unavailable: conservative two-consecutive-1m-breach approximation
                if breach and soft_pending: exit_px=ent*(1+SL_PRICE); reason='soft_stop_1m_confirm'
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
                gross=(ent-exit_px)*qty; fee=exit_px*qty*FEE_SIDE; pnl=gross-fee-pos['open_fees']; eq+=gross-fee
                closed.append({'close_ts':ts,'open_ts':pos['open_ts'],'layers':pos['layers'],'pnl_net':pnl,'reason':reason})
                if reason in ('catastrophic','soft_stop_1m_confirm'): stop_times.append(ts)
                pos=None; soft_pending=False
                if mode=='NORMAL' and eq<=peak*(1-DD_RED): mode='REDUCED'
        if eq>peak: peak=eq
        if mode=='REDUCED' and eq>=peak*(1-DD_REC): mode='NORMAL'
        # causal CAUTION: 24h realized-equity loss <= -3% OR >=3 stops in last 6h
        equity_hist.append(eq)
        ret24=0.0
        if len(equity_hist)>1440:
            ref=float(equity_hist[-1441])
            ret24=(eq/ref-1.0) if ref>0 else 0.0
        stop_times=[x for x in stop_times if x>=ts-pd.Timedelta(hours=6)]
        stop6=len(stop_times)
        caution=bool(ret24<=-0.03 or stop6>=3)
        # single-shot pause; re-arm only after cluster clears
        if stop6 < 3:
            stop_cluster_armed=True
        if stop6 >= 3 and stop_cluster_armed:
            pause_until=max(pause_until,ts+pd.Timedelta(minutes=PAUSE_MIN))
            stop_cluster_armed=False
        # new 3m model event becomes candidate at eligible_ts
        if ts in score_map:
            sc=score_map[ts]
            if sc>=BASE_TH: candidate={'score':sc,'born':ts,'expires':ts+pd.Timedelta(minutes=1)}
        # confirmation: candidate valid current/next 1m; previous closed green + current red reversal >=.01%
        if candidate is not None and ts<=candidate['expires']:
            prev=d.iloc[i-1]; prev_green=float(prev.close)>float(prev.open)
            rev=(op-cl)/op if op>0 else 0.0
            confirmed=prev_green and cl<op and rev>=0.0001
            if confirmed:
                layers=0 if pos is None else pos['layers']; can=True; sc=candidate['score']
                if (ts-last_entry_ts)<pd.Timedelta(minutes=COOLDOWN_MIN): can=False
                if layers==0 and ts<pause_until: can=False; blocked_pause+=1
                if layers==0 and exh is not None and np.isfinite(r.flow60) and float(r.flow60)<=exh: can=False; blocked_exh+=1
                if layers==1:
                    req=L2_C if caution else L2_N
                    if sc<req: can=False
                    extreme=np.isfinite(r.flow15) and np.isfinite(r.volz) and float(r.flow15)>=.10 and float(r.volz)>=2.0
                    if extreme: can=False; blocked_l2+=1
                if can and layers<2:
                    # confirmation known at minute close -> execute next minute open, causal/conservative
                    er=d.iloc[i+1]; px=float(er.open); l1,add,cap=NORMAL if mode=='NORMAL' else REDUCED
                    pct=l1 if layers==0 else add; current=0 if pos is None else pos['notional']; notional=min(eq*pct,max(0,eq*cap-current))
                    if notional>=10:
                        qty=notional/px; fee=notional*FEE_SIDE; eq-=fee
                        if pos is None:
                            pos={'qty':qty,'avg':px,'notional':notional,'layers':1,'open_fees':fee,'open_ts':er.ts,'best_roi':0.,'armed':False}
                        else:
                            nq=pos['qty']+qty; pos['avg']=(pos['avg']*pos['qty']+px*qty)/nq; pos['qty']=nq; pos['notional']+=notional; pos['layers']+=1; pos['open_fees']+=fee; pos['best_roi']=0.; pos['armed']=False
                        entries.append({'ts':er.ts,'layer':pos['layers'],'score':sc,'flow15':r.flow15,'flow60':r.flow60,'volz':r.volz,'price':px,'notional':notional})
                        last_entry_ts=ts
                candidate=None
        elif candidate is not None and ts>candidate['expires']:
            candidate=None
        mtm=eq if pos is None else eq+(pos['avg']-cl)*pos['qty']
        curve.append({'ts':ts,'equity':eq,'mtm':mtm,'layers':0 if pos is None else pos['layers']})
    E=pd.DataFrame(entries); C=pd.DataFrame(closed); Q=pd.DataFrame(curve); M=monthly(Q,C)
    p=C.pnl_net.to_numpy(float); gp=p[p>0].sum(); gl=-p[p<0].sum(); mtm=Q.mtm.to_numpy(float); pk=np.maximum.accumulate(mtm); dd=np.min(mtm/pk-1)*100
    daily=Q.assign(day=pd.to_datetime(Q.ts,utc=True).dt.floor('D')).groupby('day').equity.last().pct_change().fillna(0)*100
    neg=M[M.return_pct<0]
    summary={'name':name,'return_pct':(eq/START-1)*100,'pf':gp/gl if gl>0 else None,'maxdd_pct':dd,'entries':len(E),'closed':len(C),'neg_months':len(neg),'worst_month_pct':float(M.return_pct.min()),'median_month_pct':float(M.return_pct.median()),'worst_day_pct':float(daily.min()),'blocked_exh':blocked_exh,'blocked_l2_flow':blocked_l2,'blocked_pause':blocked_pause}
    return E,C,Q,M,summary

def main():
    d=load1m(); s=load_scores(); end=min(d.ts.max(),s.eligible_ts.max()+pd.Timedelta(minutes=2)); start=end-pd.Timedelta(days=365)
    d=d[(d.ts>=start)&(d.ts<=end)].reset_index(drop=True); s=s[(s.eligible_ts>=start)&(s.eligible_ts<=end)].reset_index(drop=True)
    print('RANGE',d.ts.min(),d.ts.max(),'1m',len(d),'scores',len(s),'gaps',int((d.ts.diff().dropna()!=pd.Timedelta(minutes=1)).sum()),flush=True)
    for name,exh in [('FLOW_F_ARCHFIX_1M',None),('FLOW_F_EXH20_ARCHFIX_1M',-.20)]:
        E,C,Q,M,S=run(d,s,exh=exh,name=name); sub=OUT/name; sub.mkdir(parents=True,exist_ok=True)
        E.to_csv(sub/'entries.csv',index=False); C.to_csv(sub/'closed_positions.csv',index=False); Q.to_csv(sub/'equity_curve.csv',index=False); M.to_csv(sub/'monthly.csv',index=False)
        print('SUMMARY',json.dumps(S,sort_keys=True),flush=True); print(M.to_string(index=False),flush=True)
if __name__=='__main__': main()
