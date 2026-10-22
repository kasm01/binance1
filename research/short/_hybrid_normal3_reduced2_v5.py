#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_INJ_WF_SCORES_V4.csv'
OUT=ROOT/'research/short/results/SHORT1_INJ_WF_HYBRID_N3_R2_V5'
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
REDUCED=(0.50,0.10,0.60)
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

def run_replay(d: pd.DataFrame):
    eq=START_EQUITY; peak=eq; mode='NORMAL'; pos=None; last_entry_i=-10**9
    entries=[]; closed=[]; curve=[]
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
                  'equity_after':eq,'mode_at_close':mode
                })
                pos=None; peak=max(peak,eq)
        peak=max(peak,eq)
        dd=(eq/peak-1.0) if peak else 0.0
        if mode=='NORMAL' and dd<=-DD_REDUCE: mode='REDUCED'
        elif mode=='REDUCED' and dd>=-DD_RECOVER: mode='NORMAL'
        score=float(row.wf_score) if np.isfinite(row.wf_score) else np.nan
        threshold=float(row.wf_threshold) if np.isfinite(row.wf_threshold) else np.nan
        can_signal=np.isfinite(score) and np.isfinite(threshold) and score>=threshold and (i-last_entry_i)>=COOLDOWN_BARS
        layers=0 if pos is None else pos['layers']
        max_layers_now=MAX_LAYERS if mode=='NORMAL' else 2
        if can_signal and layers<max_layers_now:
            ei=i+1; px=float(d.iloc[ei].open)
            l1,add,cap=NORMAL if mode=='NORMAL' else REDUCED
            pct=l1 if layers==0 else add
            target=eq*pct; current=0.0 if pos is None else pos['notional']; max_notional=eq*cap
            notional=min(target,max(0.0,max_notional-current))
            if notional>=10 and px>0:
                qty=notional/px; fee_open=notional*(FEE_RT/2); eq-=fee_open
                if pos is None:
                    pos={'qty':qty,'avg_entry':px,'notional':notional,'layers':1,'best_roi':0.0,'armed':False,
                         'soft_pending':False,'open_i':ei,'open_ts':d.iloc[ei].ts,'max_adverse_pct':0.0,
                         'max_favorable_pct':0.0,'ever_negative':False,'recovered_after_negative':False,'open_fees':fee_open}
                else:
                    nq=pos['qty']+qty
                    pos['avg_entry']=(pos['avg_entry']*pos['qty']+px*qty)/nq
                    pos['qty']=nq; pos['notional']+=notional; pos['layers']+=1
                    pos['best_roi']=0.0; pos['armed']=False; pos['soft_pending']=False; pos['open_fees']+=fee_open
                entries.append({'ts':d.iloc[ei].ts,'layer':pos['layers'],'score':score,'threshold':threshold,
                                'price':px,'notional':notional,'mode':mode,'equity_after_fee':eq})
                last_entry_i=i
        mtm=eq
        if pos is not None: mtm+=(pos['avg_entry']-cl)*pos['qty']
        curve.append({'ts':row.ts,'equity':eq,'mtm':mtm,'mode':mode,'layers':0 if pos is None else pos['layers']})
    return pd.DataFrame(entries),pd.DataFrame(closed),pd.DataFrame(curve),eq,pos

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

def main():
    d=pd.read_csv(SRC)
    d['ts']=pd.to_datetime(d.ts,utc=True)
    end=d.ts.max(); start=end-pd.Timedelta(days=365)
    d=d[d.ts>=start].reset_index(drop=True)
    e,c,curve,eq,open_pos=run_replay(d)
    c=enrich_loss_reversals(d,c)
    e.to_csv(OUT/'entries.csv',index=False); c.to_csv(OUT/'closed_positions.csv',index=False); curve.to_csv(OUT/'equity_curve.csv',index=False)
    daily=period_table(c,'D'); monthly=period_table(c,'M')
    daily.to_csv(OUT/'daily.csv',index=False); monthly.to_csv(OUT/'monthly.csv',index=False)

    # Exact two half-year windows within the same one-year test.
    mid=start+pd.Timedelta(days=182.5)
    halves=[]
    for label,a,b in [('H1',start,mid),('H2',mid,end+pd.Timedelta(seconds=1))]:
        g=c[(pd.to_datetime(c.close_ts,utc=True)>=a)&(pd.to_datetime(c.close_ts,utc=True)<b)]
        halves.append({'period':label,'start':str(a),'end':str(b),**summarize_pnl(g),
                       'net_return_on_1000_pct':float(g.pnl_net.sum()/START_EQUITY*100) if len(g) else 0.0})
    half=pd.DataFrame(halves); half.to_csv(OUT/'half_year.csv',index=False)

    losses=c[c.pnl_net<0].copy(); wins=c[c.pnl_net>0].copy()
    neg_then_win=wins[wins.ever_negative.astype(bool)].copy()
    recovered=losses[losses.recovered_to_entry==True].copy()
    loss_recovery={
      'loss_positions':int(len(losses)),
      'soft_stop_losses':int((losses.reason=='soft_stop_confirmed').sum()),
      'catastrophic_losses':int((losses.reason=='catastrophic').sum()),
      'recovered_to_original_entry_within_72h':int(len(recovered)),
      'recovery_rate_72h':safe_div(len(recovered),len(losses)),
      'time_to_entry_recovery':duration_stats(recovered.minutes_to_entry_recovery),
      'recovered_within_1h':int((recovered.minutes_to_entry_recovery<=60).sum()),
      'recovered_within_6h':int((recovered.minutes_to_entry_recovery<=360).sum()),
      'recovered_within_24h':int((recovered.minutes_to_entry_recovery<=1440).sum()),
      'recovered_within_72h':int((recovered.minutes_to_entry_recovery<=4320).sum()),
    }
    for h in POST_HORIZONS:
        col=f'max_reversal_{h}_pct'; s=pd.to_numeric(losses[col],errors='coerce').dropna()*100
        loss_recovery[f'{h}_reversal_from_stop_pct']={'mean':float(s.mean()) if len(s) else None,
            'median':float(s.median()) if len(s) else None,'p90':float(s.quantile(.90)) if len(s) else None,'max':float(s.max()) if len(s) else None}
    negwin={
      'count':int(len(neg_then_win)), 'share_of_wins':safe_div(len(neg_then_win),len(wins)),
      'duration':duration_stats(neg_then_win.duration_min),
      'max_adverse_price_pct':{
        'mean':float(neg_then_win.max_adverse_pct.mean()*100) if len(neg_then_win) else None,
        'median':float(neg_then_win.max_adverse_pct.median()*100) if len(neg_then_win) else None,
        'p90':float(neg_then_win.max_adverse_pct.quantile(.90)*100) if len(neg_then_win) else None,
        'max':float(neg_then_win.max_adverse_pct.max()*100) if len(neg_then_win) else None,
      }
    }
    curve['ts']=pd.to_datetime(curve.ts,utc=True)
    curve['date']=curve.ts.dt.strftime('%Y-%m-%d')
    deq=curve.groupby('date',sort=True).agg(end_equity=('equity','last'),end_mtm=('mtm','last')).reset_index()
    deq['prev_equity']=deq.end_equity.shift(1).fillna(START_EQUITY)
    deq['realized_delta_usdt']=deq.end_equity-deq.prev_equity
    deq['realized_return_pct']=deq.realized_delta_usdt/deq.prev_equity*100
    deq.to_csv(OUT/'daily_equity.csv',index=False)
    daily_summary={
      'calendar_days':365,
      'profitable_days':int((deq.realized_delta_usdt>1e-9).sum()),
      'losing_days':int((deq.realized_delta_usdt<-1e-9).sum()),
      'flat_days':int(365-(deq.realized_delta_usdt.abs()>1e-9).sum()),
      'avg_daily_realized_pct':float(deq.realized_return_pct.mean()),
      'median_daily_realized_pct':float(deq.realized_return_pct.median()),
      'best_day':deq.loc[deq.realized_return_pct.idxmax(),['date','realized_return_pct','realized_delta_usdt']].to_dict(),
      'worst_day':deq.loc[deq.realized_return_pct.idxmin(),['date','realized_return_pct','realized_delta_usdt']].to_dict(),
    }
    mtm=curve.mtm.astype(float); peak=mtm.cummax(); maxdd=float(((mtm/peak)-1).min()*100)
    open_unrealized=0.0
    if isinstance(open_pos,dict):
        last=float(d.iloc[-1].close); open_unrealized=(open_pos['avg_entry']-last)*open_pos['qty']-last*open_pos['qty']*(FEE_RT/2)
    total=summarize_pnl(c)
    total.update({'start':str(start),'end':str(end),'start_equity':START_EQUITY,'final_realized_equity':float(eq),
                  'realized_return_pct':float((eq/START_EQUITY-1)*100),'open_unrealized_usdt':float(open_unrealized),
                  'total_with_open_return_pct':float(((eq+open_unrealized)/START_EQUITY-1)*100),'max_mtm_dd_pct':maxdd,
                  'entries':int(len(e)),'layer1':int((e.layer==1).sum()) if len(e) else 0,
                  'layer2':int((e.layer==2).sum()) if len(e) else 0,'layer3':int((e.layer==3).sum()) if len(e) else 0,
                  'normal_entries':int((e['mode']=='NORMAL').sum()) if len(e) else 0,'reduced_entries':int((e['mode']=='REDUCED').sum()) if len(e) else 0,
                  'close_reasons':{str(k):int(v) for k,v in c.reason.value_counts().items()} if len(c) else {}})
    report={'method':'walk_forward_oos_BASE_3layer','resolution':'3m; 30s stop confirmation approximated conservatively by repeated 3m breach',
            'config':{'leverage':LEV,'tp_arm_roi':TP_ARM,'giveback_roi':GIVEBACK,'soft_stop_price_pct':SL_PRICE,
                      'catastrophic_price_pct':CATA_PRICE,'max_layers':MAX_LAYERS,'normal':'100%+50%+50%',
                      'reduced':'50%+10%; layer3 OFF','dd_governor':'-8% reduce / -4% recover','time_exit':False,'fee_roundtrip':FEE_RT},
            'total':total,'daily_summary':daily_summary,'loss_recovery':loss_recovery,'negative_then_winner':negwin,
            'monthly':monthly.to_dict('records'),'half_year':half.to_dict('records')}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False))
    print(json.dumps(report,indent=2,default=str,allow_nan=False))

if __name__=='__main__': main()
