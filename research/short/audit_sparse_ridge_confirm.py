#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,pandas as pd
ROOT=Path('/home/kasm920/binance1');sys.path.insert(0,str(ROOT))
from research.short.short_economic_features import SHORT_ECON_FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
SRC=ROOT/'research/short/results/SHORT_CANDIDATE_LIFECYCLE_FEATURES.csv';OUT=ROOT/'research/short/results'
KS=[8,12,16,24,48];FRACS=[.2,.3,.5]

def fit_select(X,y,k):
    sc=StandardScaler().fit(X);Xs=sc.transform(X);base=Ridge(alpha=10.0).fit(Xs,y);idx=np.argsort(np.abs(base.coef_))[::-1][:k]
    sc2=StandardScaler().fit(X[:,idx]);m=Ridge(alpha=10.0).fit(sc2.transform(X[:,idx]),y)
    return idx,sc2,m

def pred(obj,X):
    idx,sc,m=obj;return m.predict(sc.transform(X[:,idx]))
def met(y):
    y=np.asarray(y,float);n=len(y)
    if not n:return dict(n=0,bps=np.nan,pf=np.nan,win=np.nan)
    p=y[y>0].sum();q=-y[y<0].sum();return dict(n=n,bps=float(y.mean()*1e4),pf=float(p/max(q,1e-12)),win=float((y>0).mean()))

def main():
    d=pd.read_csv(SRC).dropna(subset=SHORT_ECON_FEATURES+['net_return_pct']).sort_values('ts').reset_index(drop=True)
    X=d[SHORT_ECON_FEATURES].to_numpy(float);y=d.net_return_pct.to_numpy(float);conf=d.state.eq('CONFIRM').to_numpy();n=len(d);hs=int(n*.8);dev=np.arange(hs);hold=np.arange(hs,n);edges=np.linspace(0,hs,6,dtype=int)
    rec=[]
    for k in KS:
      for fq in FRACS:
        fs=[]
        for seg in [2,3,4]:
          tr=np.arange(0,edges[seg-1]);va=np.arange(edges[seg-1],edges[seg]);te=np.arange(edges[seg],edges[seg+1])
          obj=fit_select(X[tr],y[tr],k);pv=pred(obj,X[va]);pt=pred(obj,X[te]);th=float(np.quantile(pv,1-fq));z=(pt>=th)&conf[te];mm=met(y[te][z]);fs.append(mm)
        N=sum(x['n'] for x in fs);pos=sum((x['bps']>0) for x in fs if x['n']);bps=float(np.average([x['bps'] for x in fs if x['n']],weights=[x['n'] for x in fs if x['n']])) if N else np.nan;worst=min([x['bps'] for x in fs if x['n']],default=np.nan)
        rec.append(dict(k=k,frac=fq,wf_n=N,wf_posfold=pos,wf_bps=bps,wf_worst=worst))
    C=pd.DataFrame(rec).sort_values(['wf_posfold','wf_worst','wf_bps','wf_n'],ascending=[False,False,False,False]);E=C[(C.wf_posfold>=2)&(C.wf_n>=15)];ch=E.iloc[0] if len(E) else C.iloc[0]
    k=int(ch.k);fq=float(ch.frac);obj=fit_select(X[dev],y[dev],k);pdv=pred(obj,X[dev]);ph=pred(obj,X[hold]);th=float(np.quantile(pdv,1-fq));z=(ph>=th)&conf[hold];hm=met(y[hold][z]);base=met(y[hold][conf[hold]])
    idx=obj[0];selected=[SHORT_ECON_FEATURES[i] for i in idx]
    print('CHOICES\n',C.to_string(index=False));print('\nSELECT',json.dumps(dict(k=k,frac=fq,features=selected),indent=2));print('HOLD_BASE',base);print('HOLD_SELECTED',hm,'threshold',th)
    C.to_csv(OUT/'SHORT_SPARSE_RIDGE_CHOICES.csv',index=False)
    pd.DataFrame({'rank':range(1,len(selected)+1),'feature':selected}).to_csv(OUT/'SHORT_SPARSE_RIDGE_FEATURES.csv',index=False)
    (OUT/'SHORT_SPARSE_RIDGE_RESULT.json').write_text(json.dumps(dict(k=k,frac=fq,features=selected,threshold=th,hold_base=base,hold_selected=hm),indent=2))
if __name__=='__main__':main()
