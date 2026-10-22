#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import redis, requests

ROOT=Path('/home/kasm920/binance1')
OUT=ROOT/'research/short/results'/os.getenv('SHORT1_V31_OUT_NAME','SHORT1_V30_SHADOW_OOS_V31'); OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'research/short/data/short1_v30_shadow_oos_v31'; DATA.mkdir(parents=True,exist_ok=True)
STREAM=os.getenv('SHORT_SHADOW_STREAM','telemetry:short1:inj:model_shadow_v3')
SYMBOL='INJUSDT'; START_EQ=1000.; LEV=3.; FEE_SIDE=.0004; TP_ARM=.0400; GIVEBACK=.0024; SL_PRICE=.0400; CATA=.0600
DD_RED=.08; DD_REC=.04; BASE_TH=.530; HARD_TH=.540; COOLDOWN_MIN=3; PAUSE_MIN=90; EXH=-.20; REENTRY_MIN=15
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']

def shadow_events(r):
    rows=[]
    for sid,f in reversed(r.xrevrange(STREAM,max='+',min='-',count=20000)):
        try:
            o=json.loads((f or {}).get('json') or '{}')
            if str(o.get('symbol') or '').upper()!=SYMBOL: continue
            rows.append({
                'stream_id':sid,'event_id':str(o.get('event_id') or ''),'source_ts':float(o.get('ts') or 0),
                'event_minute':pd.to_datetime(int(float(o.get('ts') or 0)//60)*60,unit='s',utc=True),
                'score':float(o.get('economic_probability') or 0),'confirmation_state':str(o.get('confirmation_state') or ''),
            })
        except Exception: pass
    df=pd.DataFrame(rows).sort_values('source_ts').drop_duplicates('event_id').reset_index(drop=True)
    override=os.getenv('SHORT1_V31_SCORE_OVERRIDE_CSV','').strip()
    if override:
        z=pd.read_csv(override)[['event_id','fresh_score']].copy()
        df=df.merge(z,on='event_id',how='left')
        df['score']=pd.to_numeric(df['fresh_score'],errors='coerce').fillna(df['score'])
        df=df.drop(columns=['fresh_score'])
    return df

def fetch_1m(start_ms,end_ms):
    cache=DATA/'INJUSDT_1m.csv'
    if cache.exists():
        d=pd.read_csv(cache)
        if len(d):
            mn=int(pd.to_numeric(d.open_time).min()); mx=int(pd.to_numeric(d.open_time).max())
            if mn<=start_ms and mx>=end_ms-60000: return d
    ses=requests.Session(); out=[]; cur=start_ms; nreq=0
    while cur<=end_ms:
        rr=ses.get('https://fapi.binance.com/fapi/v1/klines',params={'symbol':SYMBOL,'interval':'1m','startTime':cur,'endTime':end_ms,'limit':1500},timeout=10); rr.raise_for_status(); batch=rr.json(); nreq+=1
        if not batch: break
        out.extend(batch); last=int(batch[-1][0]); cur=last+60000
        if len(batch)<1500: break
        time.sleep(.05)
    d=pd.DataFrame(out,columns=COLS).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
    d.to_csv(cache,index=False); print('FETCH_1M',len(d),'requests',nreq,flush=True); return d

def fetch_oi(start_ms,end_ms):
    cache=DATA/'INJUSDT_oi5m.csv'
    if cache.exists():
        d=pd.read_csv(cache)
        if len(d):
            mn=int(pd.to_numeric(d.timestamp).min()); mx=int(pd.to_numeric(d.timestamp).max())
            if mn<=start_ms and mx>=end_ms-300000: return d
    # Binance OI history returns the most recent rows inside a window. Page backwards
    # from end_ms so the full multi-day range is covered rather than only the last 500.
    ses=requests.Session(); out=[]; cur_end=end_ms; nreq=0
    while cur_end>=start_ms:
        rr=ses.get('https://fapi.binance.com/futures/data/openInterestHist',params={'symbol':SYMBOL,'period':'5m','startTime':start_ms,'endTime':cur_end,'limit':500},timeout=10); rr.raise_for_status(); batch=rr.json(); nreq+=1
        if not batch: break
        out.extend(batch); first=min(int(x.get('timestamp') or 0) for x in batch)
        if first<=start_ms or len(batch)<500: break
        cur_end=first-1
        time.sleep(.05)
    d=pd.DataFrame(out).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    d.to_csv(cache,index=False); print('FETCH_OI',len(d),'requests',nreq,flush=True); return d

def prepare(d,oi):
    for c in ['open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce').astype('int64')
    for c in ['open','high','low','close','quote_volume','taker_buy_quote_volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True)
    net=2*d.taker_buy_quote_volume-d.quote_volume
    d['flow15']=net.rolling(15,min_periods=15).sum()/d.quote_volume.rolling(15,min_periods=15).sum()
    d['flow60']=net.rolling(60,min_periods=60).sum()/d.quote_volume.rolling(60,min_periods=60).sum()
    d['ret15']=d.close.pct_change(15)
    # Strictly completed before event minute.
    d['flow15_completed']=d.flow15.shift(1); d['flow60_completed']=d.flow60.shift(1); d['ret15_completed']=d.ret15.shift(1)
    oi=oi.copy(); oi['timestamp']=pd.to_numeric(oi.timestamp,errors='coerce').astype('int64'); oi['oi_ts']=pd.to_datetime(oi.timestamp,unit='ms',utc=True)
    oi['sumOpenInterest']=pd.to_numeric(oi.sumOpenInterest,errors='coerce'); oi['oi15']=oi.sumOpenInterest.pct_change(3)
    d['oi_lookup_ts']=d.ts-pd.Timedelta(nanoseconds=1)
    d=pd.merge_asof(d.sort_values('oi_lookup_ts'),oi[['oi_ts','oi15']],left_on='oi_lookup_ts',right_on='oi_ts',direction='backward',tolerance=pd.Timedelta('5m'))
    return d.sort_values('ts').reset_index(drop=True)

def metrics(C,Q):
    p=C.pnl_net.to_numpy(float) if len(C) else np.array([]); gp=p[p>0].sum() if len(p) else 0.; gl=-p[p<0].sum() if len(p) else 0.
    mtm=Q.mtm.to_numpy(float); peak=np.maximum.accumulate(mtm); dd=float(np.min(mtm/peak-1)*100) if len(mtm) else 0.
    daily=Q.set_index('ts').mtm.resample('1D').last().ffill(); dr=daily.pct_change().dropna()*100
    return {'pf':(float(gp/gl) if gl>0 else None),'maxdd_pct':dd,'worst_day_pct':(float(dr.min()) if len(dr) else None),'positive_days':int((dr>0).sum()),'negative_days':int((dr<0).sum())}

def run(d,ev):
    event_map={t:g.to_dict('records') for t,g in ev.groupby('event_minute')}
    eq=START_EQ; peak=eq; mode='NORMAL'; pos=None; soft_pending=False; last_entry=pd.Timestamp('1970-01-01',tz='UTC'); stop_reentry_until=last_entry; pause_until=last_entry
    stop_times=[]; stop_cluster_armed=True; equity_hist=[]; E=[]; C=[]; Q=[]; decisions=[]
    blocked={'low_score':0,'confirm':0,'soft_context':0,'flow60':0,'position':0,'cooldown':0,'reentry':0,'pause':0}
    for i in range(1,len(d)-1):
        r=d.iloc[i]; ts=r.ts; hi=float(r.high); lo=float(r.low); cl=float(r.close)
        if pos is not None:
            ent=pos['avg']; qty=pos['qty']; exit_px=None; reason=None
            if hi>=ent*(1+CATA): exit_px=ent*(1+CATA); reason='catastrophic'
            else:
                breach=hi>=ent*(1+SL_PRICE)
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
                C.append({'close_ts':ts,'open_ts':pos['open_ts'],'pnl_net':pnl,'reason':reason,'score':pos['score'],'band':pos['band']})
                if reason in ('catastrophic','soft_stop_1m_confirm'):
                    stop_times.append(ts); stop_reentry_until=max(stop_reentry_until,ts+pd.Timedelta(minutes=REENTRY_MIN))
                pos=None; soft_pending=False
                if mode=='NORMAL' and eq<=peak*(1-DD_RED): mode='REDUCED'
        if eq>peak: peak=eq
        if mode=='REDUCED' and eq>=peak*(1-DD_REC): mode='NORMAL'
        equity_hist.append(eq); ret24=0.
        if len(equity_hist)>1440:
            ref=float(equity_hist[-1441]); ret24=(eq/ref-1) if ref>0 else 0.
        stop_times=[x for x in stop_times if x>=ts-pd.Timedelta(hours=6)]; stop6=len(stop_times)
        if stop6<3: stop_cluster_armed=True
        if stop6>=3 and stop_cluster_armed:
            pause_until=max(pause_until,ts+pd.Timedelta(minutes=PAUSE_MIN)); stop_cluster_armed=False
        for e in event_map.get(ts,[]):
            sc=float(e['score']); conf=str(e['confirmation_state'])
            reason='READY'; allow=True
            if sc<BASE_TH: allow=False; reason='model_score_low'; blocked['low_score']+=1
            elif conf!='CONFIRM': allow=False; reason='entry_confirmation_not_ready'; blocked['confirm']+=1
            elif pos is not None: allow=False; reason='position_exists'; blocked['position']+=1
            elif (ts-last_entry)<pd.Timedelta(minutes=COOLDOWN_MIN): allow=False; reason='symbol_cooldown'; blocked['cooldown']+=1
            elif ts<stop_reentry_until: allow=False; reason='stop_reentry_cooldown'; blocked['reentry']+=1
            elif ts<pause_until: allow=False; reason='risk_pause'; blocked['pause']+=1
            flow60=float(r.flow60_completed) if np.isfinite(r.flow60_completed) else np.nan
            ret15=float(r.ret15_completed) if np.isfinite(r.ret15_completed) else 0.; flow15=float(r.flow15_completed) if np.isfinite(r.flow15_completed) else 0.; oi15=float(r.oi15) if np.isfinite(r.oi15) else 0.
            confirmed_build=(oi15>0 and ret15!=0 and flow15!=0 and ((ret15>0)==(flow15>0)))
            build_dir='long' if confirmed_build and ret15>0 else ('short' if confirmed_build else 'none')
            if allow and np.isfinite(flow60) and flow60<=EXH: allow=False; reason='short_money_flow_exhaustion'; blocked['flow60']+=1
            if allow and sc<HARD_TH and not (confirmed_build and build_dir=='short'):
                allow=False; reason='soft_short_build_not_confirmed'; blocked['soft_context']+=1
            decisions.append({'ts':ts,'score':sc,'confirmation':conf,'allow':allow,'reason':reason,'ret15':ret15,'flow15':flow15,'flow60':flow60,'oi15':oi15,'build_direction':build_dir})
            if allow:
                er=d.iloc[i+1]; px=float(er.open); context_mult=.5 if confirmed_build else 1.; base_mult=.5 if mode=='REDUCED' else 1.; mult=base_mult*context_mult; notional=eq*mult
                if notional>=10:
                    qty=notional/px; fee=notional*FEE_SIDE; eq-=fee; pos={'qty':qty,'avg':px,'notional':notional,'open_fees':fee,'open_ts':er.ts,'best_roi':0.,'armed':False,'score':sc,'band':'SOFT' if sc<HARD_TH else 'HARD'}
                    E.append({'ts':er.ts,'score':sc,'band':pos['band'],'price':px,'notional':notional,'context_multiplier':context_mult,'risk_mode':mode,'build_direction':build_dir,'flow60':flow60}); last_entry=ts
        mtm=eq if pos is None else eq+(pos['avg']-cl)*pos['qty']; Q.append({'ts':ts,'equity':eq,'mtm':mtm,'position_open':pos is not None})
    E=pd.DataFrame(E); C=pd.DataFrame(C); Q=pd.DataFrame(Q); D=pd.DataFrame(decisions)
    final_mtm=float(Q.mtm.iloc[-1]); m=metrics(C,Q); p=C.pnl_net.to_numpy(float) if len(C) else np.array([])
    summary={'source_events':len(ev),'confirm_events':int((ev.confirmation_state=='CONFIRM').sum()),'entries':len(E),'closed':len(C),'wins':int((p>0).sum()) if len(p) else 0,'losses':int((p<0).sum()) if len(p) else 0,'win_rate':float((p>0).mean()) if len(p) else None,'realized_return_pct':(eq/START_EQ-1)*100,'mtm_return_pct':(final_mtm/START_EQ-1)*100,'final_equity':eq,'final_mtm':final_mtm,'blocked':blocked,**m}
    return E,C,Q,D,summary

def main():
    r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
    ev=shadow_events(r); assert len(ev)>0
    # Need 90m prehistory for flow/OI and future through current completed minute.
    start_ms=int(ev.event_minute.min().timestamp()*1000)-2*3600_000
    end_ms=int(pd.Timestamp.now(tz='UTC').floor('min').timestamp()*1000)-60_000
    d=fetch_1m(start_ms,end_ms); oi=fetch_oi(start_ms,end_ms); d=prepare(d,oi)
    # Trim lifecycle to first shadow minute onward, retaining computed rolling context.
    run_start=ev.event_minute.min(); d_run=d[(d.ts>=run_start)&(d.ts<=pd.to_datetime(end_ms,unit='ms',utc=True))].reset_index(drop=True)
    E,C,Q,D,S=run(d_run,ev)
    ev.to_csv(OUT/'source_events.csv',index=False); E.to_csv(OUT/'entries.csv',index=False); C.to_csv(OUT/'closed_positions.csv',index=False); Q.to_csv(OUT/'equity_curve.csv',index=False); D.to_csv(OUT/'decisions.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(S,indent=2,allow_nan=False))
    print('RANGE',run_start,d_run.ts.max(),'1m',len(d_run),'events',len(ev),'confirm',int((ev.confirmation_state=='CONFIRM').sum()))
    print('SUMMARY',json.dumps(S,sort_keys=True))
    if len(E): print('\nENTRIES\n',E.to_string(index=False))
    if len(C): print('\nCLOSED\n',C.to_string(index=False))
    print('\nDECISION REASONS\n',D.reason.value_counts().to_string())

if __name__=='__main__': main()
