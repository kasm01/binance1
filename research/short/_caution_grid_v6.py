#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_INJ_WF_SCORES_V4.csv'
OUT=ROOT/'research/short/results/SHORT1_THR_0p54_L3M005'
OUT.mkdir(parents=True,exist_ok=True)

START_EQUITY=1000.0
LEV=3.0
FEE_RT=0.0008
TP_ARM=0.0400
GIVEBACK=0.0024
SL_PRICE=0.0299
CATA_PRICE=0.035
MAX_LAYERS=3
DD_REDUCE=0.08
DD_RECOVER=0.04
NORMAL=(1.00,0.50,2.00)
REDUCED=(0.50,0.10,0.70)
COOLDOWN_BARS=1

# Post-stop reversal horizons in 3m bars.
POST_HORIZONS={'1h':20,'6h':120,'24h':480,'72h':1440}

def q(x,p):
    s=pd.Series(x,dtype='float64').dropna()
    return None if s.empty else float(s.quantile(p))

def safe_div(a,b):
    return None if not b else float(a/b)

def summarize_pnl(c: pd.DataFrame):
    if c.empty: return {}
    pos=c[c.pnl_net>0]; neg=c[c.pnl_net<0]
    gp=float(pos.pnl_net.sum()); gl=float(-neg.pnl_net.sum())
    return {
      'closed_positions':int(len(c)), 'wins':int(len(pos)), 'losses':int(len(neg)),
      'win_rate':float(len(pos)/len(c)), 'loss_rate':float(len(neg)/len(c)),
      'gross_profit_usdt':gp, 'gross_loss_usdt':gl, 'net_closed_pnl_usdt':float(c.pnl_net.sum()),
      'profit_factor':safe_div(gp,gl), 'avg_win_usdt':float(pos.pnl_net.mean()) if len(pos) else None,
      'avg_loss_usdt':float(neg.pnl_net.mean()) if len(neg) else None,
      'median_win_usdt':float(pos.pnl_net.median()) if len(pos) else None,
      'median_loss_usdt':float(neg.pnl_net.median()) if len(neg) else None,
    }

def run_replay(d: pd.DataFrame, *, name='BASE', caution_l3_margin=0.005, caution_l3_notional_mult=1.0, caution_enabled=False):
    eq=START_EQUITY; peak=eq; mode='NORMAL'; pos=None; last_entry_i=-10**9
    entries=[]; closed=[]; curve=[]; equity_hist=[]; stop_close_bars=[]
    caution_bars=0; caution_entries=0; caution_l3_entries=0
    for i in range(len(d)-1):
        row=d.iloc[i]; hi=float(row.high); lo=float(row.low); cl=float(row.close)
        # lifecycle first
        if pos is not None:
            ent=pos['avg_entry']; qty=pos['qty']
            adverse=(hi/ent)-1.0; favorable=1.0-(lo/ent)
            pos['max_adverse_pct']=max(pos['max_adverse_pct'],adverse)
            pos['max_favorable_pct']=max(pos['max_favorable_pct'],favorable)
            if cl>ent: pos['ever_negative']=True
            if cl<=ent and pos['ever_negative']: pos['recovered_after_negative']=True
            exit_px=None; reason=None
            cata_px=ent*(1+CATA_PRICE); soft_px=ent*(1+SL_PRICE)
            if hi>=cata_px:
                exit_px=cata_px; reason='catastrophic'
            else:
                breached=hi>=soft_px
                if breached and pos.get('soft_pending',False):
                    exit_px=soft_px; reason='soft_stop_confirmed'
                pos['soft_pending']=bool(breached)
                if not breached: pos['soft_pending']=False
                if exit_px is None and pos.get('armed',False):
                    trig=pos['best_roi']-GIVEBACK
                    trig_px=ent*(1-trig/LEV)
                    if hi>=trig_px:
                        exit_px=trig_px; reason='tp_giveback'
                if exit_px is None:
                    roi=((ent-lo)/ent)*LEV
                    if roi>pos['best_roi']: pos['best_roi']=roi
                    if pos['best_roi']>=TP_ARM: pos['armed']=True
            if exit_px is not None:
                gross=(ent-exit_px)*qty
                exit_fee=exit_px*qty*(FEE_RT/2)
                pnl_cash=gross-exit_fee
                pnl_net=pnl_cash-float(pos.get('open_fees') or 0.0)
                eq+=pnl_cash
                closed.append({
                  'open_i':pos['open_i'],'close_i':i,'open_ts':pos['open_ts'],'close_ts':row.ts,
                  'entry_price':ent,'exit_price':exit_px,'qty':qty,'layers':pos['layers'],'reason':reason,
                  'pnl_net':pnl_net,'pnl_pct_equity':pnl_net/START_EQUITY,'duration_min':(i-pos['open_i']+1)*3,
                  'best_roi':pos['best_roi'],'max_adverse_pct':pos['max_adverse_pct'],
                  'max_favorable_pct':pos['max_favorable_pct'],'ever_negative':bool(pos['ever_negative']),
                  'recovered_after_negative':bool(pos['recovered_after_negative']),
                  'equity_after':eq,'mode_at_close':mode,'caution_at_close':bool(pos.get('caution_at_open',False))
                })
                if reason in ('soft_stop_confirmed','catastrophic'):
                    stop_close_bars.append(i)
                pos=None; peak=max(peak,eq)
        peak=max(peak,eq)
        dd=(eq/peak-1.0) if peak else 0.0
        if mode=='NORMAL' and dd<=-DD_REDUCE: mode='REDUCED'
        elif mode=='REDUCED' and dd>=-DD_RECOVER: mode='NORMAL'

        # causal CAUTION state: 24h realized-equity loss OR >=3 stops in last 6h
        equity_hist.append(eq)
        ret24=0.0
        if i>=480:
            ref=float(equity_hist[i-480])
            ret24=(eq/ref-1.0) if ref>0 else 0.0
        stop_close_bars=[j for j in stop_close_bars if j>=i-120]
        stop6=len(stop_close_bars)
        caution=bool(caution_enabled and (ret24<=-0.03 or stop6>=3))
        if caution: caution_bars+=1

        score=float(row.wf_score) if np.isfinite(row.wf_score) else np.nan
        threshold=0.54
        can_signal=np.isfinite(score) and score>=threshold and (i-last_entry_i)>=COOLDOWN_BARS
        layers=0 if pos is None else pos['layers']
        l3_margin=0.005
        if layers==2 and caution:
            l3_margin=float(caution_l3_margin)
        if layers==2 and can_signal:
            can_signal = score >= (threshold + l3_margin)
        if can_signal and layers<MAX_LAYERS:
            ei=i+1; px=float(d.iloc[ei].open)
            l1,add,cap=NORMAL if mode=='NORMAL' else REDUCED
            pct=l1 if layers==0 else add
            if layers==2 and caution:
                pct *= float(caution_l3_notional_mult)
            target=eq*pct; current=0.0 if pos is None else pos['notional']; max_notional=eq*cap
            notional=min(target,max(0.0,max_notional-current))
            if notional>=10 and px>0:
                qty=notional/px; fee_open=notional*(FEE_RT/2); eq-=fee_open
                if caution: caution_entries+=1
                if layers==2 and caution: caution_l3_entries+=1
                if pos is None:
                    pos={'qty':qty,'avg_entry':px,'notional':notional,'layers':1,'best_roi':0.0,'armed':False,
                         'soft_pending':False,'open_i':ei,'open_ts':d.iloc[ei].ts,'max_adverse_pct':0.0,
                         'max_favorable_pct':0.0,'ever_negative':False,'recovered_after_negative':False,'open_fees':fee_open,
                         'caution_at_open':caution}
                else:
                    nq=pos['qty']+qty
                    pos['avg_entry']=(pos['avg_entry']*pos['qty']+px*qty)/nq
                    pos['qty']=nq; pos['notional']+=notional; pos['layers']+=1
                    pos['best_roi']=0.0; pos['armed']=False; pos['soft_pending']=False; pos['open_fees']+=fee_open
                    pos['caution_at_open']=bool(pos.get('caution_at_open',False) or caution)
                entries.append({'ts':d.iloc[ei].ts,'layer':pos['layers'],'score':score,'threshold':threshold,
                                'l3_margin':l3_margin,'price':px,'notional':notional,'mode':mode,'caution':caution,
                                'ret24':ret24,'stop6':stop6,'equity_after_fee':eq})
                last_entry_i=i
        mtm=eq
        if pos is not None: mtm+=(pos['avg_entry']-cl)*pos['qty']
        curve.append({'ts':row.ts,'equity':eq,'mtm':mtm,'mode':mode,'layers':0 if pos is None else pos['layers'],
                      'caution':caution,'ret24':ret24,'stop6':stop6})
    meta={'name':name,'caution_bars':caution_bars,'caution_entries':caution_entries,'caution_l3_entries':caution_l3_entries}
    return pd.DataFrame(entries),pd.DataFrame(closed),pd.DataFrame(curve),eq,pos,meta

def enrich_loss_reversals(d: pd.DataFrame, c: pd.DataFrame):
    if c.empty: return c
    c=c.copy()
    for col in ['recovered_to_entry','minutes_to_entry_recovery','reversal_to_entry_pct','max_reversal_1h_pct','max_reversal_6h_pct','max_reversal_24h_pct','max_reversal_72h_pct','minutes_to_max_reversal_24h']:
        c[col]=np.nan
    loss_idx=c.index[c.pnl_net<0]
    for j in loss_idx:
        r=c.loc[j]; ci=int(r.close_i); exit_px=float(r.exit_price); ent=float(r.entry_price)
        max_bars=max(POST_HORIZONS.values())
        fut=d.iloc[ci+1:min(len(d),ci+1+max_bars)].copy()
        if fut.empty: continue
        lows=pd.to_numeric(fut.low,errors='coerce').to_numpy(float)
        # For a SHORT, recovery after a stop means price falls back to/below original aggregate entry.
        hit=np.where(lows<=ent)[0]
        recovered=len(hit)>0
        c.at[j,'recovered_to_entry']=bool(recovered)
        c.at[j,'reversal_to_entry_pct']=max(0.0,(exit_px-ent)/exit_px)
        if recovered:
            c.at[j,'minutes_to_entry_recovery']=float((int(hit[0])+1)*3)
        for name,bars in POST_HORIZONS.items():
            arr=lows[:min(len(lows),bars)]
            if len(arr):
                rev=max(0.0,(exit_px-float(np.nanmin(arr)))/exit_px)
                c.at[j,f'max_reversal_{name}_pct']=rev
        arr=lows[:min(len(lows),POST_HORIZONS['24h'])]
        if len(arr):
            mi=int(np.nanargmin(arr)); c.at[j,'minutes_to_max_reversal_24h']=float((mi+1)*3)
    return c

def period_table(c: pd.DataFrame, freq: str):
    if c.empty: return pd.DataFrame()
    x=c.copy(); x['close_ts']=pd.to_datetime(x.close_ts,utc=True)
    if freq=='D': x['period']=x.close_ts.dt.strftime('%Y-%m-%d')
    elif freq=='M': x['period']=x.close_ts.dt.to_period('M').astype(str)
    elif freq=='6M':
        # Rolling calendar half-years anchored to Jan/Jul for readability.
        y=x.close_ts.dt.year; h=np.where(x.close_ts.dt.month<=6,'H1','H2')
        x['period']=y.astype(str)+'-'+pd.Series(h,index=x.index)
    out=[]
    for p,g in x.groupby('period',sort=True):
        m=summarize_pnl(g)
        out.append({'period':p,**m,'net_return_on_1000_pct':float(g.pnl_net.sum()/START_EQUITY*100),
                    'avg_duration_min':float(g.duration_min.mean()),'median_duration_min':float(g.duration_min.median())})
    return pd.DataFrame(out)

def duration_stats(s):
    s=pd.Series(s,dtype='float64').dropna()
    if s.empty: return {'n':0}
    return {'n':int(len(s)),'mean_min':float(s.mean()),'median_min':float(s.median()),'p90_min':float(s.quantile(.90)),'max_min':float(s.max())}

def monthly_equity_table(curve, closed):
    cv=curve.copy(); cv['ts']=pd.to_datetime(cv.ts,utc=True); cv['month']=cv.ts.dt.to_period('M').astype(str)
    cc=closed.copy(); cc['close_ts']=pd.to_datetime(cc.close_ts,utc=True); cc['month']=cc.close_ts.dt.to_period('M').astype(str)
    rows=[]
    prev=START_EQUITY
    for month,g in cv.groupby('month',sort=True):
        endeq=float(g.equity.iloc[-1]); ret=(endeq/prev-1)*100 if prev>0 else 0
        cg=cc[cc.month==month]
        gp=float(cg.loc[cg.pnl_net>0,'pnl_net'].sum()); gl=float(-cg.loc[cg.pnl_net<0,'pnl_net'].sum())
        rows.append({'month':month,'start_equity':prev,'end_equity':endeq,'return_pct':ret,'closed':len(cg),
                     'wins':int((cg.pnl_net>0).sum()),'losses':int((cg.pnl_net<0).sum()),
                     'pf':gp/gl if gl>0 else None,'l3_closes':int((cg.layers==3).sum()),
                     'l3_pnl':float(cg.loc[cg.layers==3,'pnl_net'].sum()),
                     'stop_pnl':float(cg.loc[cg.reason.isin(['soft_stop_confirmed','catastrophic']),'pnl_net'].sum()),
                     'tp_pnl':float(cg.loc[cg.reason=='tp_giveback','pnl_net'].sum())})
        prev=endeq
    return pd.DataFrame(rows)

def main():
    d=pd.read_csv(SRC); d['ts']=pd.to_datetime(d.ts,utc=True)
    end=d.ts.max(); start=end-pd.Timedelta(days=365); d=d[d.ts>=start].reset_index(drop=True)
    variants=[
      ('BASE',False,0.005,1.0),
      ('CAUTION_A_M010',True,0.010,1.0),
      ('CAUTION_B_M015',True,0.015,1.0),
      ('CAUTION_C_M010_HALF_L3',True,0.010,0.5),
    ]
    root=ROOT/'research/short/results/SHORT1_CAUTION_GRID_V6'; root.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name,en,margin,mult in variants:
        e,c,curve,eq,pos,meta=run_replay(d,name=name,caution_enabled=en,caution_l3_margin=margin,caution_l3_notional_mult=mult)
        sub=root/name; sub.mkdir(parents=True,exist_ok=True)
        e.to_csv(sub/'entries.csv',index=False); c.to_csv(sub/'closed_positions.csv',index=False); curve.to_csv(sub/'equity_curve.csv',index=False)
        mon=monthly_equity_table(curve,c); mon.to_csv(sub/'monthly.csv',index=False)
        p=c.pnl_net.to_numpy(float); gp=float(p[p>0].sum()); gl=float(-p[p<0].sum())
        mtm=curve.mtm.to_numpy(float); peak=np.maximum.accumulate(mtm); maxdd=float(np.min(mtm/peak-1)*100)
        daily=curve.copy(); daily['date']=pd.to_datetime(daily.ts,utc=True).dt.floor('D'); dg=daily.groupby('date').equity.last(); dr=dg.pct_change().fillna(0)*100
        neg=mon[mon.return_pct<0]
        summary={'name':name,'annual_realized_pct':(eq/START_EQUITY-1)*100,'pf':gp/gl if gl>0 else None,'maxdd_pct':maxdd,
                 'closed':len(c),'entries':len(e),'entries_day':len(e)/365,'neg_months':len(neg),
                 'worst_month':None if neg.empty else str(neg.loc[neg.return_pct.idxmin(),'month']),
                 'worst_month_pct':0 if neg.empty else float(neg.return_pct.min()),'median_month_pct':float(mon.return_pct.median()),
                 'worst_day_pct':float(dr.min()),'best_day_pct':float(dr.max()),**meta}
        summaries.append(summary)
        print('SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
        print(mon[['month','return_pct','pf','closed','l3_closes','l3_pnl','stop_pnl','tp_pnl']].to_string(index=False),flush=True)
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)

if __name__=='__main__': main()
