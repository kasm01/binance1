#!/usr/bin/env python3
from pathlib import Path
import sys, json, numpy as np, pandas as pd
ROOT=Path('/home/kasm920/binance1'); sys.path.insert(0,str(ROOT))
from research.short.short_economic_features import SHORT_ECON_FEATURES
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
SRC=ROOT/'research/short/results/SHORT_CANDIDATE_LIFECYCLE_FEATURES.csv'; OUT=ROOT/'research/short/results'
FRACS=[.10,.20,.30,.50]; MODELS=['ridge','hgb']

def fit(kind,X,y,seed):
    if kind=='ridge': return Pipeline([('sc',StandardScaler()),('m',Ridge(alpha=10.0))]).fit(X,y)
    return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10.0,learning_rate=.04,random_state=seed).fit(X,y)

def met(y):
    y=np.asarray(y,float); n=len(y)
    if not n:return dict(n=0,bps=np.nan,win=np.nan,pf=np.nan)
    p=y[y>0].sum(); q=-y[y<0].sum(); return dict(n=n,bps=float(y.mean()*1e4),win=float((y>0).mean()),pf=float(p/max(q,1e-12)))

def run_scope(name,d):
    X=d[SHORT_ECON_FEATURES].to_numpy(float); y=d.net_return_pct.to_numpy(float); conf=(d.state.to_numpy()=='CONFIRM'); n=len(d)
    hs=int(n*.80); dev=np.arange(hs); hold=np.arange(hs,n); edges=np.linspace(0,hs,6,dtype=int)
    folds=[]; choices=[]
    for kind in MODELS:
        kr=[]
        for seg in [2,3,4]:
            tr=np.arange(0,edges[seg-1]); va=np.arange(edges[seg-1],edges[seg]); te=np.arange(edges[seg],edges[seg+1])
            m=fit(kind,X[tr],y[tr],2000+seg); pv=m.predict(X[va]); pt=m.predict(X[te])
            for fq in FRACS:
                th=float(np.quantile(pv,1-fq)); z=(pt>=th)&conf[te]; mm=met(y[te][z]); row=dict(scope=name,model=kind,fold=seg,frac=fq,threshold=th,**mm); folds.append(row); kr.append(row)
        K=pd.DataFrame(kr)
        for fq,g in K.groupby('frac'):
            gg=g[g.n>0]; N=int(gg.n.sum()); pos=int((gg.bps>0).sum()); b=float(np.average(gg.bps,weights=gg.n)) if N else np.nan; worst=float(gg.bps.min()) if len(gg) else np.nan
            choices.append(dict(scope=name,model=kind,frac=float(fq),wf_n=N,wf_posfold=pos,wf_bps=b,wf_worst=worst))
    C=pd.DataFrame(choices).sort_values(['wf_posfold','wf_bps','wf_n'],ascending=[False,False,False]); E=C[(C.wf_posfold>=2)&(C.wf_n>=10)]; ch=(E.iloc[0] if len(E) else C.iloc[0])
    kind=str(ch.model); fq=float(ch.frac); m=fit(kind,X[dev],y[dev],9999); pdv=m.predict(X[dev]); ph=m.predict(X[hold]); th=float(np.quantile(pdv,1-fq)); z=(ph>=th)&conf[hold]
    hm=met(y[hold][z]); base=met(y[hold][conf[hold]])
    res=dict(scope=name,rows=n,model=kind,frac=fq,wf_posfold=int(ch.wf_posfold),wf_n=int(ch.wf_n),wf_bps=float(ch.wf_bps),wf_worst=float(ch.wf_worst),threshold=th,hold_confirm_base_n=base['n'],hold_confirm_base_bps=base['bps'],hold_confirm_base_pf=base['pf'],**{f'hold_{k}':v for k,v in hm.items()})
    return folds,choices,res

def main():
    d=pd.read_csv(SRC).dropna(subset=SHORT_ECON_FEATURES+['net_return_pct']).sort_values('ts').reset_index(drop=True)
    fs=[]; cs=[]; rs=[]
    for name,g in [('ALL',d)]+[(s,x.sort_values('ts').reset_index(drop=True)) for s,x in d.groupby('symbol')]:
        if len(g)<100: continue
        f,c,r=run_scope(name,g); fs+=f;cs+=c;rs.append(r); print('RESULT',json.dumps(r,sort_keys=True),flush=True)
    pd.DataFrame(fs).to_csv(OUT/'SHORT_MODEL_CONFIRM_WF.csv',index=False); pd.DataFrame(cs).to_csv(OUT/'SHORT_MODEL_CONFIRM_CHOICES.csv',index=False); pd.DataFrame(rs).to_csv(OUT/'SHORT_MODEL_CONFIRM_HOLDOUT.csv',index=False)
    print('\nFINAL\n',pd.DataFrame(rs).to_string(index=False))
if __name__=='__main__': main()
