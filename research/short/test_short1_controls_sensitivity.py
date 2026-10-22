import importlib.util, pandas as pd, numpy as np, json
from collections import deque
spec=importlib.util.spec_from_file_location('m','research/short/replay_short1_combined_c600_v30.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=m.load1m(); s=m.load_scores()
end=min(d.ts.max(),s.eligible_ts.max()+pd.Timedelta(minutes=2)); start=end-pd.Timedelta(days=365)
d=d[(d.ts>=start)&(d.ts<=end)].reset_index(drop=True); s=s[(s.eligible_ts>=start)&(s.eligible_ts<=end)].reset_index(drop=True)
N=len(d); ts=pd.to_datetime(d.ts,utc=True); ns=ts.astype('int64').to_numpy()
O=d.open.to_numpy(float); H=d.high.to_numpy(float); L=d.low.to_numpy(float); CL=d.close.to_numpy(float)
F15=d.flow15_completed.to_numpy(float); F60=d.flow60.to_numpy(float); OI=d.oi15.to_numpy(float); R15=d.ret15_completed.to_numpy(float)
score_map=dict(zip(s.eligible_ts.astype('int64'),s.wf_score)); SCORE=np.array([score_map.get(x,np.nan) for x in ns],float)
MIN=60_000_000_000; HOUR=60*MIN; DAY=24*HOUR

def run(name,band_mult=1.0,flowpos_mult=1.0,day_loss_cap=None,stop_n=3,stop_pause_min=90):
    eq=m.START; peak=eq; mode=0; pos=None; last_entry=-10**30; pause_until=-10**30; reentry_until=-10**30
    stops=deque(); cluster_armed=True; candidate=None; cand_exp=-1; soft_pending=False
    cur_day=-1; day_start=eq; day_block=False; entries=[]; closes=[]; curve=np.empty(N-2,float)
    for i in range(1,N-1):
        t=ns[i]; day=t//DAY
        if day!=cur_day: cur_day=day; day_start=eq; day_block=False
        if pos is not None:
            ent,qty=pos['avg'],pos['qty']; exit_px=None; reason=None
            if H[i]>=ent*(1+m.CATA): exit_px=ent*(1+m.CATA); reason='catastrophic'
            else:
                breach=H[i]>=ent*(1+m.SL_PRICE)
                if breach and soft_pending: exit_px=ent*(1+m.SL_PRICE); reason='soft_stop_1m_confirm'
                soft_pending=bool(breach)
                if not breach: soft_pending=False
                if exit_px is None and pos['armed']:
                    trig=pos['best']-m.GIVEBACK; px=ent*(1-trig/m.LEV)
                    if H[i]>=px: exit_px=px; reason='tp_giveback'
                if exit_px is None:
                    roi=((ent-L[i])/ent)*m.LEV
                    if roi>pos['best']: pos['best']=roi
                    if pos['best']>=m.TP_ARM: pos['armed']=True
            if exit_px is not None:
                gross=(ent-exit_px)*qty; fee=exit_px*qty*m.FEE_SIDE; pnl=gross-fee-pos['open_fee']; eq+=gross-fee
                closes.append((t,pos['open_ns'],pnl,reason,pos['score']))
                if reason in ('catastrophic','soft_stop_1m_confirm'):
                    stops.append(t); reentry_until=max(reentry_until,t+15*MIN)
                pos=None; soft_pending=False
                if mode==0 and eq<=peak*(1-m.DD_RED): mode=1
                if day_loss_cap is not None and day_start>0 and eq/day_start-1<=-abs(day_loss_cap): day_block=True
        if eq>peak: peak=eq
        if mode==1 and eq>=peak*(1-m.DD_REC): mode=0
        cutoff=t-6*HOUR
        while stops and stops[0]<cutoff: stops.popleft()
        if len(stops)<stop_n: cluster_armed=True
        if len(stops)>=stop_n and cluster_armed:
            pause_until=max(pause_until,t+stop_pause_min*MIN); cluster_armed=False
        sc=SCORE[i]
        if np.isfinite(sc) and sc>=m.BASE_TH: candidate=float(sc); cand_exp=t+MIN
        if candidate is not None and t<=cand_exp:
            confirmed=CL[i-1]>O[i-1] and CL[i]<O[i] and ((O[i]-CL[i])/O[i] if O[i]>0 else 0)>=.0001
            if confirmed:
                sc=candidate; can=True
                if sc<m.HARD_TH:
                    ret15=R15[i] if np.isfinite(R15[i]) else 0.; f15=F15[i] if np.isfinite(F15[i]) else 0.; oi=OI[i] if np.isfinite(OI[i]) else 0.
                    can=ret15<0 and f15<0 and oi>0
                if t-last_entry<3*MIN or pos is not None or t<reentry_until or t<pause_until or day_block: can=False
                if np.isfinite(F60[i]) and F60[i]<=-.20: can=False
                oi_reduce=False
                if can:
                    ret15=R15[i] if np.isfinite(R15[i]) else 0.; f15=F15[i] if np.isfinite(F15[i]) else 0.; oi=OI[i] if np.isfinite(OI[i]) else 0.
                    oi_reduce=(oi>0 and ret15!=0 and f15!=0 and ((ret15>0)==(f15>0)))
                if can:
                    px=O[i+1]; base_pct=1.0 if mode==0 else .5; cap=2.0 if mode==0 else .7; mult=.5 if oi_reduce else 1.0
                    if .55<=sc<.575: mult*=band_mult
                    if (F15[i] if np.isfinite(F15[i]) else 0.)>=0: mult*=flowpos_mult
                    notional=min(eq*base_pct*mult,eq*cap)
                    if notional>=10:
                        qty=notional/px; fee=notional*m.FEE_SIDE; eq-=fee
                        pos={'avg':px,'qty':qty,'open_fee':fee,'open_ns':ns[i+1],'best':0.,'armed':False,'score':sc}
                        entries.append((ns[i+1],sc,notional)); last_entry=t
                candidate=None
        elif candidate is not None and t>cand_exp: candidate=None
        curve[i-1]=eq if pos is None else eq+(pos['avg']-CL[i])*pos['qty']
    C=pd.DataFrame(closes,columns=['close_ns','open_ns','pnl','reason','score']); E=pd.DataFrame(entries,columns=['open_ns','score','notional'])
    p=C.pnl.to_numpy(float); gp=p[p>0].sum(); gl=-p[p<0].sum(); q=curve; pk=np.maximum.accumulate(q); dd=np.min(q/pk-1)*100
    C['month']=pd.to_datetime(C.close_ns,utc=True).dt.strftime('%Y-%m'); months=['2025-11','2026-02','2026-05','2026-06','2026-07','2026-09']
    mp={mo:float(C.loc[C.month==mo,'pnl'].sum()) for mo in months}
    qdf=pd.DataFrame({'ts':ts.iloc[1:-1].to_numpy(),'eq':curve}); qdf['month']=pd.to_datetime(qdf.ts,utc=True).dt.strftime('%Y-%m')
    me=qdf.groupby('month').eq.last(); prev=me.shift(1); prev.iloc[0]=m.START; mr=(me/prev-1)*100; mrt={mo:float(mr.get(mo,np.nan)) for mo in months}
    return {'name':name,'return_pct':(eq/m.START-1)*100,'pf':gp/gl,'maxdd_pct':float(dd),'entries':len(E),'closed':len(C),'months_pnl':mp,'months_ret':mrt}

variants=[
('BASE',1,1,None,3,90),
('F25_D3',1,.25,.03,3,90),('F50_D3',1,.50,.03,3,90),('F75_D3',1,.75,.03,3,90),
('F50_D25',1,.50,.025,3,90),('F50_D35',1,.50,.035,3,90),('F50_D4',1,.50,.04,3,90),
('B25_F50_D3',.25,.50,.03,3,90),('B50_F50_D3',.50,.50,.03,3,90),('B75_F50_D3',.75,.50,.03,3,90),
('B50_F25_D3',.50,.25,.03,3,90),('B50_F75_D3',.50,.75,.03,3,90),
('B50_F50_D25',.50,.50,.025,3,90),('B50_F50_D35',.50,.50,.035,3,90),('B50_F50_D4',.50,.50,.04,3,90)
]
for v in variants:
    print(json.dumps(run(v[0],v[1],v[2],v[3],v[4],v[5]),sort_keys=True),flush=True)
