#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, os, sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from features.pipeline import make_matrix, FEATURE_SCHEMA_20

SYMS=[x.strip().upper() for x in os.getenv('SHORT_AUDIT_SYMBOLS','LINKUSDT,AVAXUSDT').split(',') if x.strip()]
SRC_DIR=Path(os.getenv('SHORT_AUDIT_DATA_DIR','/home/kasm920/binance1-pro/data/offline_cache'))
OUT=Path('research/short/results'); OUT.mkdir(parents=True,exist_ok=True)
HORIZON=int(os.getenv('SHORT_AUDIT_LABEL_HORIZON','3'))
LABEL_THR=float(os.getenv('SHORT_AUDIT_LABEL_THR','0.0005'))
FEE_RT=float(os.getenv('SHORT_AUDIT_ROUNDTRIP_FEE','0.0008'))
LEVERAGE=float(os.getenv('SHORT_AUDIT_LEVERAGE','2'))
ROI_TP=float(os.getenv('SHORT_AUDIT_ROI_TP','0.0055'))
ROI_STOP=abs(float(os.getenv('SHORT_AUDIT_ROI_STOP','-0.0060')))
TP_PX=ROI_TP/max(LEVERAGE,1e-9)
SL_PX=ROI_STOP/max(LEVERAGE,1e-9)
BARRIER_BARS=int(os.getenv('SHORT_AUDIT_BARRIER_BARS','6'))  # 18m on 3m bars
FRACS=[.0025,.005,.01,.02,.05]


def labels(close):
    c=np.asarray(close,float); fut=np.roll(c,-HORIZON); ret=fut/np.maximum(c,1e-12)-1
    y=(ret>LABEL_THR).astype(int); y[-HORIZON:]=0
    return y

def barrier_returns(d, idx):
    # short entry at next 3m open after completed decision bar
    o=d.open.to_numpy(float); h=d.high.to_numpy(float); l=d.low.to_numpy(float); c=d.close.to_numpy(float)
    out=np.empty(len(idx),float); outcome=[]
    for j,r in enumerate(idx):
        ent_i=r+1
        if ent_i+BARRIER_BARS>len(d): out[j]=np.nan; outcome.append('na'); continue
        e=o[ent_i]; res=None
        # Conservative same-bar tie: adverse stop wins when both barriers touched.
        for k in range(ent_i,min(len(d),ent_i+BARRIER_BARS)):
            stop=(h[k]/e-1)>=SL_PX
            tp=(1-l[k]/e)>=TP_PX
            if stop: res=-SL_PX-FEE_RT; outcome.append('sl'); break
            if tp: res=TP_PX-FEE_RT; outcome.append('tp'); break
        if res is None:
            ex=c[min(len(d)-1,ent_i+BARRIER_BARS-1)]
            res=(1-ex/e)-FEE_RT; outcome.append('timeout')
        out[j]=res
    return out,np.asarray(outcome,object)

def fit_model(X,y,seed):
    # Same family/contract as production trainer, deterministic temporal fit.
    cnt=np.bincount(y,minlength=2); cnt=np.maximum(cnt,1)
    sw=np.where(y==1,0.5/cnt[1],0.5/cnt[0])
    m=Pipeline([('scaler',StandardScaler()),('sgd',SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,max_iter=80,tol=1e-3,shuffle=False,random_state=seed,n_jobs=-1))])
    m.fit(X,y,sgd__sample_weight=sw)
    return m

def metrics(ret):
    r=np.asarray(ret,float); r=r[np.isfinite(r)]
    if not len(r): return {'n':0,'net_bps':None,'win':None,'pf':None,'sum_ret':None}
    pos=r[r>0].sum(); neg=-r[r<0].sum()
    return {'n':int(len(r)),'net_bps':float(r.mean()*1e4),'win':float((r>0).mean()),'pf':float(pos/max(neg,1e-12)),'sum_ret':float(r.sum())}

summ=[]; details=[]
for sym in SYMS:
    p=SRC_DIR/f'{sym}_3m_360d.csv'
    d=pd.read_csv(p)
    X=make_matrix(d,schema=FEATURE_SCHEMA_20)
    X=np.asarray(X,float)
    y=labels(pd.to_numeric(d.close,errors='coerce').to_numpy(float))
    valid=np.isfinite(X).all(1)
    valid[-max(HORIZON,BARRIER_BARS+1):]=False
    rows=np.flatnonzero(valid); Xv=X[rows]; yv=y[rows]
    rets,outs=barrier_returns(d,rows)
    ok=np.isfinite(rets); rows=rows[ok]; Xv=Xv[ok]; yv=yv[ok]; rets=rets[ok]; outs=outs[ok]
    n=len(rows); hold_start=int(n*5/6)
    dev=np.arange(hold_start); hold=np.arange(hold_start,n)
    edges=np.linspace(0,hold_start,6,dtype=int)
    foldrows=[]
    # walk-forward: train expanding, validation block selects quantile, next block tests
    for testseg in [2,3,4]:
        tr=np.arange(0,edges[testseg-1]); va=np.arange(edges[testseg-1],edges[testseg]); te=np.arange(edges[testseg],edges[testseg+1])
        m=fit_model(Xv[tr],yv[tr],4200+testseg)
        pv=1-m.predict_proba(Xv[va])[:,1]; pt=1-m.predict_proba(Xv[te])[:,1]
        auc=float(roc_auc_score(yv[te],1-pt)) if len(np.unique(yv[te]))>1 else .5
        for frac in FRACS:
            thr=float(np.quantile(pv,1-frac)); sel=pt>=thr
            mm=metrics(rets[te][sel]);
            foldrows.append({'symbol':sym,'fold':testseg,'frac':frac,'threshold':thr,'auc_buy_test':auc,**mm})
    F=pd.DataFrame(foldrows)
    # choose fraction on WF robustness: positive folds first, weighted net second; require samples
    choices=[]
    for frac,g in F.groupby('frac'):
        gg=g[g.n>0]; total=int(gg.n.sum()); posfold=int((gg.net_bps>0).sum()); worst=float(gg.net_bps.min()) if len(gg) else np.nan
        wnet=float(np.average(gg.net_bps,weights=gg.n)) if total else np.nan
        choices.append({'frac':frac,'wf_n':total,'positive_folds':posfold,'wf_net_bps':wnet,'wf_worst_bps':worst})
    C=pd.DataFrame(choices).sort_values(['positive_folds','wf_net_bps','wf_n'],ascending=[False,False,False])
    eligible=C[(C.positive_folds>=2)&(C.wf_n>=30)]
    chosen=(eligible.iloc[0] if len(eligible) else C.iloc[0]); frac=float(chosen.frac)
    # lock frac, train full dev, threshold from dev score quantile, untouched holdout
    m=fit_model(Xv[dev],yv[dev],9999)
    pdv=1-m.predict_proba(Xv[dev])[:,1]; ph=1-m.predict_proba(Xv[hold])[:,1]
    thr=float(np.quantile(pdv,1-frac)); sel=ph>=thr
    hm=metrics(rets[hold][sel]); houts=outs[hold][sel]
    hold_auc=float(roc_auc_score(yv[hold],1-ph)) if len(np.unique(yv[hold]))>1 else .5
    tp=int((houts=='tp').sum()); sl=int((houts=='sl').sum()); timeout=int((houts=='timeout').sum())
    summary={'symbol':sym,'rows':n,'dev_rows':len(dev),'hold_rows':len(hold),'chosen_frac':frac,
             'wf_positive_folds':int(chosen.positive_folds),'wf_n':int(chosen.wf_n),'wf_net_bps':float(chosen.wf_net_bps),'wf_worst_bps':float(chosen.wf_worst_bps),
             'hold_auc_buy':hold_auc,'threshold_short_score':thr,**{f'hold_{k}':v for k,v in hm.items()},'hold_tp':tp,'hold_sl':sl,'hold_timeout':timeout,
             'tp_px':TP_PX,'sl_px':SL_PX,'fee_rt':FEE_RT,'barrier_bars':BARRIER_BARS}
    summ.append(summary); details.extend(foldrows)
    print('SUMMARY',json.dumps(summary,sort_keys=True),flush=True)

pd.DataFrame(details).to_csv(OUT/'SHORT_HYBRID_TAIL_3M_WF.csv',index=False)
pd.DataFrame(summ).to_csv(OUT/'SHORT_HYBRID_TAIL_3M_HOLDOUT.csv',index=False)
(OUT/'SHORT_HYBRID_TAIL_3M_HOLDOUT.json').write_text(json.dumps(summ,indent=2))
print('\nFINAL')
print(pd.DataFrame(summ).to_string(index=False))
