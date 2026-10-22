from pathlib import Path
import json, math
import numpy as np, pandas as pd
P=Path('research/short/results/SHORT1_INJ_WF_SCORES_V4.csv')
FEE=.0008; LEV=3.; TP=.04; GB=.0024; SL=.0299; CATA=.035; DD_RED=.08; DD_REC=.04
NORMAL=(1.,.5,2.); REDUCED=(.5,.1,.7); START=1000.; BAD_UP=.03
WIN=3360; MINP=1440

d=pd.read_csv(P); d['ts']=pd.to_datetime(d.ts,utc=True)
# past-only rolling score quantile, reset by model fold
ad=np.full(len(d),np.nan)
for fid,g in d.groupby('wf_fold',sort=False):
    idx=g.index.to_numpy(); frac=float(g.wf_frac.iloc[0]); past=g.wf_score.shift(1)
    rq=past.rolling(WIN,min_periods=MINP).quantile(1-frac).to_numpy(float)
    base=g.wf_threshold.to_numpy(float); ad[idx]=np.maximum(base,np.where(np.isfinite(rq),rq,base))
d['adaptive_threshold']=ad
good=(pd.to_numeric(d.ret_240)<=0)&(pd.to_numeric(d.atr14_pct)<=pd.to_numeric(d.wf_atr90)); d['ctx_good']=good
# diagnostics by fold
print('CALIBRATION')
for fid,g in d.groupby('wf_fold'):
    b=(g.wf_score>=g.wf_threshold).mean(); a=(g.wf_score>=g.adaptive_threshold).mean(); target=float(g.wf_frac.iloc[0]); print(fid,'target',round(target,3),'static',round(b,3),'adaptive',round(a,3),'thr',round(float(g.wf_threshold.iloc[0]),4),'ad_med',round(float(g.adaptive_threshold.median()),4))

ts=d.ts.to_numpy(); op=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float); cl=d.close.to_numpy(float); sc=d.wf_score.to_numpy(float); base=d.wf_threshold.to_numpy(float); ada=d.adaptive_threshold.to_numpy(float); ctx=d.ctx_good.to_numpy(bool)

def replay(start,policy):
    s=int(d['ts'].searchsorted(pd.Timestamp(start))); eq=START; peak=eq; mode='NORMAL'; pos=None; last=-10**9; pnls=[]; reasons=[]; entries=goodn=badn=normaln=reducedn=0; maxdd=0.0
    for i in range(s,len(d)-1):
        if pos is not None:
            ent=pos['avg']; exitpx=None; reason=None
            if hi[i]>=ent*(1+CATA): exitpx=ent*(1+CATA); reason='catastrophic'
            else:
                breach=hi[i]>=ent*(1+SL)
                if breach and pos['pending']: exitpx=ent*(1+SL); reason='soft_stop_confirmed'
                pos['pending']=bool(breach)
                if not breach: pos['pending']=False
                if exitpx is None and pos['armed']:
                    trig=pos['best']-GB; px=ent*(1-trig/LEV)
                    if hi[i]>=px: exitpx=px; reason='tp_giveback'
                if exitpx is None:
                    roi=((ent-lo[i])/ent)*LEV
                    if roi>pos['best']: pos['best']=roi
                    if pos['best']>=TP: pos['armed']=True
            if exitpx is not None:
                pnl=(ent-exitpx)*pos['qty']-exitpx*pos['qty']*(FEE/2); eq+=pnl; pnls.append(pnl); reasons.append(reason); pos=None; peak=max(peak,eq)
        peak=max(peak,eq); dd=eq/peak-1
        if mode=='NORMAL' and dd<=-DD_RED: mode='REDUCED'
        elif mode=='REDUCED' and dd>=-DD_REC: mode='NORMAL'
        th=base[i] if policy.startswith('STATIC') else ada[i]
        if policy.endswith('CTX') and not ctx[i]: th+=BAD_UP
        layers=0 if pos is None else pos['layers']
        if sc[i]>=th and i-last>=1 and layers<3:
            cfg=NORMAL if mode=='NORMAL' else REDUCED; pct=cfg[0] if layers==0 else cfg[1]; cur=0 if pos is None else pos['notional']; notional=min(eq*pct,max(0.,eq*cfg[2]-cur)); px=op[i+1]
            if notional>=10 and px>0:
                q=notional/px; eq-=notional*(FEE/2); entries+=1; goodn+=int(ctx[i]); badn+=int(not ctx[i]); normaln+=int(mode=='NORMAL'); reducedn+=int(mode=='REDUCED')
                if pos is None: pos={'qty':q,'avg':px,'notional':notional,'layers':1,'best':0.,'armed':False,'pending':False}
                else:
                    nq=pos['qty']+q; pos['avg']=(pos['avg']*pos['qty']+px*q)/nq; pos['qty']=nq; pos['notional']+=notional; pos['layers']+=1; pos['best']=0.; pos['armed']=False; pos['pending']=False
                last=i
        mtm=eq+(0 if pos is None else (pos['avg']-cl[i])*pos['qty']); peak_m=max(START,peak); maxdd=min(maxdd,mtm/peak_m-1)
    unreal=0.
    if pos is not None: unreal=(pos['avg']-cl[-1])*pos['qty']-cl[-1]*pos['qty']*(FEE/2)
    p=np.asarray(pnls,float); gp=p[p>0].sum() if len(p) else 0.; gl=-p[p<0].sum() if len(p) else 0.
    from collections import Counter
    return {'policy':policy,'entries':entries,'closed':len(p),'return_pct':(eq/START-1)*100,'total_pct':((eq+unreal)/START-1)*100,'win_pct':float((p>0).mean()*100) if len(p) else None,'pf':float(gp/gl) if gl>0 else None,'max_dd_pct':maxdd*100,'good_entries':goodn,'bad_entries':badn,'normal_entries':normaln,'reduced_entries':reducedn,'reasons':dict(Counter(reasons))}
end=d.ts.iloc[-1]
for pol in ['STATIC','STATIC_CTX','ADAPT','ADAPT_CTX']:
    print('POLICY',pol)
    for name,days in [('1M',30),('6M',182),('1Y',365)]: print(name,json.dumps(replay(end-pd.Timedelta(days=days),pol),sort_keys=True))
    print('TRUE_FORWARD',json.dumps(replay(pd.Timestamp('2026-09-14T20:24:00Z'),pol),sort_keys=True))
