#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np, pandas as pd
ROOT=Path('/home/kasm920/binance1')
ONE=ROOT/'research/short/data/inj_1m'
SCORES=ROOT/'research/short/results/SHORT1_INJ_WF_SCORES_V4.csv'
OI=ROOT/'research/short/data/INJUSDT_metrics_5m_20251012_20261013.csv'
MICRO=ROOT/'research/short/results/SHORT1_V47_CANDIDATE_MICRO.csv'
OUT=ROOT/'research/short/results/SHORT1_V47_PATH'; OUT.mkdir(parents=True,exist_ok=True)
START=1000.0; LEV=3.0; FEE_SIDE=0.0004; TP_ARM=.0400; GIVEBACK=.0024; SL_PRICE=.0400; CATA=.0600
NORMAL=(1.00,.50,2.00); REDUCED=(.50,.10,.70); DD_RED=.08; DD_REC=.04
BASE_TH=.530; HARD_TH=.540; L2_N=.545; L2_C=.550; COOLDOWN_MIN=3; PAUSE_MIN=90

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
    d['ret15']=d.close.pct_change(15)
    m=pd.read_csv(OI)
    m['oi_ts']=pd.to_datetime(m['create_time'],utc=True)
    m=m.sort_values('oi_ts').reset_index(drop=True)
    m['oi15']=pd.to_numeric(m['sum_open_interest'],errors='coerce').pct_change(3)
    # Strictly prior OI snapshot: do not consume an OI record stamped exactly at the current 1m bar start.
    d['oi_lookup_ts']=d.ts-pd.Timedelta(nanoseconds=1)
    d=pd.merge_asof(d.sort_values('oi_lookup_ts'),m[['oi_ts','oi15']],left_on='oi_lookup_ts',right_on='oi_ts',direction='backward',tolerance=pd.Timedelta('5m'))
    d=d.sort_values('ts').reset_index(drop=True)
    # Live-safe context: only bars completed before the current confirmation candle.
    d['flow15_completed']=d['flow15'].shift(1)
    d['ret15_completed']=d['ret15'].shift(1)
    return d

def load_scores():
    s=pd.read_csv(SCORES); s['score_ts']=pd.to_datetime(s.ts,utc=True); s['eligible_ts']=s.score_ts+pd.Timedelta(minutes=3)
    return s[['score_ts','eligible_ts','wf_score']]

def load_micro():
    m=pd.read_csv(MICRO)
    m['confirm_ts']=pd.to_datetime(m.confirm_ts,utc=True)
    cols=['flow_5s','flow_20s','flow_30s','ret_20s','ret_30s','flow_delta_5_20','ret_delta_20_30']
    for c in cols: m[c]=pd.to_numeric(m[c],errors='coerce')
    return {r.confirm_ts:{c:getattr(r,c) for c in cols} for r in m.itertuples(index=False)}

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

def run(d,s,exh=None,name='BASE',oi_policy='NONE',context_timing='ENTRY',micro_map=None,hazard_enable=False):
    score_map={t:float(v) for t,v in zip(s.eligible_ts,s.wf_score) if np.isfinite(v)}
    eq=START; peak=eq; mode='NORMAL'; pos=None; last_entry_ts=pd.Timestamp('1970-01-01',tz='UTC'); pause_until=last_entry_ts; stop_reentry_until=last_entry_ts
    stop_times=[]; candidate=None; entries=[]; closed=[]; curve=[]; blocked_exh=blocked_l2=blocked_pause=blocked_oi=blocked_reentry=0
    soft_pending=False; equity_hist=[]; stop_cluster_armed=True
    haz_pending=haz_reject=haz_ready=haz_notapp=haz_missing=0; hazard_events=[]
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
                if reason in ('catastrophic','soft_stop_1m_confirm'):
                    stop_times.append(ts)
                    stop_reentry_until=max(stop_reentry_until,ts+pd.Timedelta(minutes=15))
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
            if sc>=BASE_TH:
                master_ret15=float(r.ret15_completed) if np.isfinite(r.ret15_completed) else 0.0
                master_oi15=float(r.oi15) if np.isfinite(r.oi15) else 0.0
                master_flow15=float(r.flow15_completed) if np.isfinite(r.flow15_completed) else 0.0
                master_confirmed_build=(master_oi15>0.0 and master_ret15!=0.0 and master_flow15!=0.0 and ((master_ret15>0)==(master_flow15>0)))
                candidate={'score':sc,'born':ts,'expires':ts+pd.Timedelta(minutes=1),'master_confirmed_build':bool(master_confirmed_build)}
        # confirmation: candidate valid current/next 1m; previous closed green + current red reversal >=.01%
        if candidate is not None and ts<=candidate['expires']:
            prev=d.iloc[i-1]; prev_green=float(prev.close)>float(prev.open)
            rev=(op-cl)/op if op>0 else 0.0
            confirmed=prev_green and cl<op and rev>=0.0001
            if confirmed:
                layers=0 if pos is None else pos['layers']; can=True; sc=candidate['score']
                if sc < HARD_TH:
                    ret15_soft=float(r.ret15_completed) if np.isfinite(r.ret15_completed) else 0.0
                    flow15_soft=float(r.flow15_completed) if np.isfinite(r.flow15_completed) else 0.0
                    oi15_soft=float(r.oi15) if np.isfinite(r.oi15) else 0.0
                    can = can and (ret15_soft < 0.0 and flow15_soft < 0.0 and oi15_soft > 0.0)
                # V47: shadow-contract hazard overlay at ENTRY CONFIRMATION.
                # NOT_APPLICABLE is fallback-to-baseline, not a rejection. Only PENDING+30s rebound rejects.
                if hazard_enable and can:
                    mm=(micro_map or {}).get(ts)
                    if mm is None:
                        haz_missing+=1
                    else:
                        f5=float(mm.get('flow_5s')) if np.isfinite(mm.get('flow_5s')) else np.nan
                        f20=float(mm.get('flow_20s')) if np.isfinite(mm.get('flow_20s')) else np.nan
                        r20=float(mm.get('ret_20s')) if np.isfinite(mm.get('ret_20s')) else np.nan
                        r30=float(mm.get('ret_30s')) if np.isfinite(mm.get('ret_30s')) else np.nan
                        fd=(f20-f5) if np.isfinite(f20) and np.isfinite(f5) else np.nan
                        confirmed20=bool(np.isfinite(r20) and r20<=-0.0001)
                        pending=bool(confirmed20 and np.isfinite(f20) and np.isfinite(fd) and f20<=-0.60 and fd<=-0.30)
                        hstate='NORMAL'
                        if not confirmed20:
                            haz_notapp+=1; hstate='NOT_APPLICABLE'
                        elif pending:
                            haz_pending+=1
                            if np.isfinite(r30) and (r30-r20)>=0.0:
                                haz_reject+=1; hstate='REJECT_CANDIDATE'; can=False
                            elif np.isfinite(r30):
                                haz_ready+=1; hstate='READY_AFTER_REVALIDATION'
                            else:
                                hstate='INSUFFICIENT'
                        hazard_events.append({'confirm_ts':ts,'score':sc,'state':hstate,'flow5':f5,'flow20':f20,'ret20':r20,'ret30':r30,'flow_delta_5_20':fd,'can_after_hazard':bool(can)})
                if (ts-last_entry_ts)<pd.Timedelta(minutes=COOLDOWN_MIN): can=False
                if layers==0 and ts<stop_reentry_until: can=False; blocked_reentry+=1
                if layers==0 and ts<pause_until: can=False; blocked_pause+=1
                if layers==0 and exh is not None and np.isfinite(r.flow60) and float(r.flow60)<=exh:
                    can=False; blocked_exh+=1
                oi_reduce=False
                if layers==0 and can and oi_policy != 'NONE':
                    if str(context_timing).upper() == 'MASTER':
                        confirmed_build=bool(candidate.get('master_confirmed_build',False))
                        new_long=False
                    else:
                        ret15=float(r.ret15_completed) if np.isfinite(r.ret15_completed) else 0.0
                        oi15=float(r.oi15) if np.isfinite(r.oi15) else 0.0
                        flow15=float(r.flow15_completed) if np.isfinite(r.flow15_completed) else 0.0
                        new_long=(ret15>0.0 and oi15>0.0 and flow15>0.0)
                        confirmed_build=(oi15>0.0 and ret15!=0.0 and flow15!=0.0 and ((ret15>0)==(flow15>0)))
                    oi_reduce=((oi_policy=='RISK50_NEW_LONG' and new_long) or (oi_policy=='RISK50_CONFIRMED_BUILD' and confirmed_build))
                    if oi_reduce: blocked_oi+=1
                if layers==1:
                    req=L2_C if caution else L2_N
                    if sc<req: can=False
                    extreme=np.isfinite(r.flow15) and np.isfinite(r.volz) and float(r.flow15)>=.10 and float(r.volz)>=2.0
                    if extreme: can=False; blocked_l2+=1
                if can and layers<1:
                    # confirmation known at minute close -> execute next minute open, causal/conservative
                    er=d.iloc[i+1]; px=float(er.open); l1,add,cap=NORMAL if mode=='NORMAL' else REDUCED
                    pct=l1 if layers==0 else add
                    if layers==0 and oi_reduce:
                        pct=float(pct)*0.50
                    current=0 if pos is None else pos['notional']; notional=min(eq*pct,max(0,eq*cap-current))
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
    summary={'name':name,'context_timing':str(context_timing).upper(),'return_pct':(eq/START-1)*100,'pf':gp/gl if gl>0 else None,'maxdd_pct':dd,'entries':len(E),'closed':len(C),'neg_months':len(neg),'worst_month_pct':float(M.return_pct.min()),'median_month_pct':float(M.return_pct.median()),'worst_day_pct':float(daily.min()),'blocked_exh':blocked_exh,'blocked_l2_flow':blocked_l2,'blocked_pause':blocked_pause,'blocked_oi':blocked_oi,'blocked_reentry':blocked_reentry,'haz_pending':haz_pending,'haz_reject':haz_reject,'haz_ready':haz_ready,'haz_notapp':haz_notapp,'haz_missing':haz_missing}
    return E,C,Q,M,summary,pd.DataFrame(hazard_events)

def period_report(Q,C,freq):
    q=Q.copy(); q['ts']=pd.to_datetime(q.ts,utc=True)
    c=C.copy();
    if len(c): c['close_ts']=pd.to_datetime(c.close_ts,utc=True)
    if freq=='D': q['period']=q.ts.dt.strftime('%Y-%m-%d'); c['period']=c.close_ts.dt.strftime('%Y-%m-%d') if len(c) else []
    elif freq=='W':
        iso=q.ts.dt.isocalendar(); q['period']=iso.year.astype(str)+'-W'+iso.week.astype(str).str.zfill(2)
        if len(c):
            ci=c.close_ts.dt.isocalendar(); c['period']=ci.year.astype(str)+'-W'+ci.week.astype(str).str.zfill(2)
    else: q['period']=q.ts.dt.strftime('%Y-%m'); c['period']=c.close_ts.dt.strftime('%Y-%m') if len(c) else []
    # use MTM endpoint return to capture open position path
    e=q.groupby('period').mtm.last(); prev=e.shift(1); prev.iloc[0]=START
    rows=[]
    for per,v in e.items():
        x=c[c.period==per] if len(c) else c; p=x.pnl_net.to_numpy(float) if len(x) else np.array([]); gp=p[p>0].sum() if len(p) else 0.; gl=-p[p<0].sum() if len(p) else 0.
        qq=q[q.period==per]; arr=qq.mtm.to_numpy(float); pk=np.maximum.accumulate(arr) if len(arr) else np.array([]); dd=float(np.min(arr/pk-1)*100) if len(arr) else np.nan
        rows.append({'period':per,'return_pct':(v/prev.loc[per]-1)*100,'closed':len(x),'pnl_net':float(p.sum()) if len(p) else 0.,'pf':gp/gl if gl>0 else np.nan,'period_maxdd_pct':dd})
    return pd.DataFrame(rows)

def deep_dive(months, baseC, hazC, baseE, hazE, H):
    out=[]
    for m in months:
        bc=baseC[pd.to_datetime(baseC.close_ts,utc=True).dt.strftime('%Y-%m')==m].copy(); hc=hazC[pd.to_datetime(hazC.close_ts,utc=True).dt.strftime('%Y-%m')==m].copy()
        be=baseE[pd.to_datetime(baseE.ts,utc=True).dt.strftime('%Y-%m')==m].copy(); he=hazE[pd.to_datetime(hazE.ts,utc=True).dt.strftime('%Y-%m')==m].copy()
        hh=H[pd.to_datetime(H.confirm_ts,utc=True).dt.strftime('%Y-%m')==m].copy() if len(H) else H
        def sm(c):
            p=c.pnl_net.to_numpy(float) if len(c) else np.array([]); gp=p[p>0].sum() if len(p) else 0.; gl=-p[p<0].sum() if len(p) else 0.
            return {'closed':len(c),'wins':int((p>0).sum()),'losses':int((p<0).sum()),'pnl':float(p.sum()) if len(p) else 0.,'pf':float(gp/gl) if gl>0 else None,'stops':int(c.reason.astype(str).str.contains('stop|catastrophic',case=False,regex=True).sum()) if len(c) else 0}
        out.append({'month':m,'baseline':sm(bc),'v47':sm(hc),'entry_delta':len(he)-len(be),'haz_pending':int((hh.state.isin(['REJECT_CANDIDATE','READY_AFTER_REVALIDATION'])).sum()) if len(hh) else 0,'haz_reject':int((hh.state=='REJECT_CANDIDATE').sum()) if len(hh) else 0,'haz_ready':int((hh.state=='READY_AFTER_REVALIDATION').sum()) if len(hh) else 0})
    return out

def main():
    d=load1m(); s=load_scores(); micro=load_micro(); end=min(d.ts.max(),s.eligible_ts.max()+pd.Timedelta(minutes=2)); start=end-pd.Timedelta(days=365)
    d=d[(d.ts>=start)&(d.ts<=end)].reset_index(drop=True); s=s[(s.eligible_ts>=start)&(s.eligible_ts<=end)].reset_index(drop=True)
    print('RANGE',d.ts.min(),d.ts.max(),'1m',len(d),'scores',len(s),'micro',len(micro),flush=True)
    B_E,B_C,B_Q,B_M,B_S,B_H=run(d,s,exh=-.20,name='BASELINE_C600',oi_policy='RISK50_CONFIRMED_BUILD',context_timing='ENTRY',micro_map=micro,hazard_enable=False)
    V_E,V_C,V_Q,V_M,V_S,V_H=run(d,s,exh=-.20,name='V47_HAZARD_PATH',oi_policy='RISK50_CONFIRMED_BUILD',context_timing='ENTRY',micro_map=micro,hazard_enable=True)
    # exact baseline parity against canonical stored replay
    canon=pd.read_csv(ROOT/'research/short/results/SHORT1_COMBINED_V30/C600/closed_positions.csv')
    canon_pnl=float(pd.to_numeric(canon.pnl_net,errors='coerce').sum())
    parity={'closed_v47baseline':len(B_C),'closed_canonical':len(canon),'pnl_v47baseline':float(B_C.pnl_net.sum()),'pnl_canonical':canon_pnl,'pnl_abs_diff':abs(float(B_C.pnl_net.sum())-canon_pnl)}
    print('BASELINE_PARITY',json.dumps(parity),flush=True)
    if len(B_C)!=len(canon) or parity['pnl_abs_diff']>1e-6: raise SystemExit('BASELINE PARITY FAILED')
    for name,E,C,Q,M,S,H in [('BASELINE',B_E,B_C,B_Q,B_M,B_S,B_H),('V47',V_E,V_C,V_Q,V_M,V_S,V_H)]:
        sub=OUT/name; sub.mkdir(parents=True,exist_ok=True); E.to_csv(sub/'entries.csv',index=False); C.to_csv(sub/'closed_positions.csv',index=False); Q.to_csv(sub/'equity_curve.csv',index=False); H.to_csv(sub/'hazard_events.csv',index=False)
        for freq,label in [('D','daily'),('W','weekly'),('M','monthly')]: period_report(Q,C,freq).to_csv(sub/f'{label}.csv',index=False)
        (sub/'summary.json').write_text(json.dumps(S,indent=2))
    months=['2026-04','2026-05','2026-07','2026-09']; dd=deep_dive(months,B_C,V_C,B_E,V_E,V_H)
    comp={'baseline':B_S,'v47':V_S,'delta_return_pct':V_S['return_pct']-B_S['return_pct'],'delta_maxdd_pct':V_S['maxdd_pct']-B_S['maxdd_pct'],'delta_pf':(V_S['pf'] or 0)-(B_S['pf'] or 0),'baseline_parity':parity,'deep_dive':dd}
    (OUT/'comparison.json').write_text(json.dumps(comp,indent=2))
    print('BASE',json.dumps(B_S,sort_keys=True)); print('V47',json.dumps(V_S,sort_keys=True)); print('COMPARE',json.dumps(comp,indent=2),flush=True)
if __name__=='__main__': main()
