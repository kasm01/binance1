#!/usr/bin/env python3
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
SRC=Path('/home/kasm920/binance1-pro/data/offline_cache');OUT=Path('research/short/results');OUT.mkdir(parents=True,exist_ok=True)
SYMS=['LINKUSDT','AVAXUSDT'];FEE=.0008;TP=.006;SL=.010;H=40;FRACS=[.0025,.005,.01,.02,.05,.10]

def feat(d):
 o=d.open.astype(float);h=d.high.astype(float);l=d.low.astype(float);c=d.close.astype(float);v=d.volume.astype(float);tb=d.taker_buy_base_volume.astype(float);r=c.pct_change();f=pd.DataFrame(index=d.index)
 for n in [1,2,3,6,12,20,60,240]:f[f'ret{n}']=c.pct_change(n)
 f['body']=(c-o)/o;f['range']=(h-l)/c;rng=(h-l).replace(0,np.nan);f['close_loc']=(c-l)/rng;f['upper_wick']=(h-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-l)/c
 for sp in [5,10,20,50,100]:
  e=c.ewm(span=sp,adjust=False).mean();f[f'e{sp}_rel']=c/e-1;f[f'e{sp}_slope3']=e/e.shift(3)-1
 pc=c.shift(1);tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1);f['atr14']=tr.rolling(14).mean()/c
 for w in [10,20,60,240]:f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r.rolling(w).std();f[f'down_frac{w}']=(r<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r.rolling(w).mean()
 buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
 for w in [3,6,12,30]:f[f'imb_mean{w}']=f.imbalance.rolling(w).mean()
 f['ret_accel']=r-r.shift(1);f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std();
 return f.replace([np.inf,-np.inf],np.nan)
def barrier(d,rows):
 o=d.open.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float);cl=d.close.to_numpy(float);e=o[rows+1];idx=(rows+1)[:,None]+np.arange(H)[None,:];sh=(hi[idx]/e[:,None]-1)>=SL;th=(1-lo[idx]/e[:,None])>=TP;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);itp=(ti<si)&(ti<H);tout=1-cl[rows+H]/e-FEE;return np.where(isl,-SL-FEE,np.where(itp,TP-FEE,tout))
def met(r):
 r=np.asarray(r);n=len(r)
 if not n:return (0,np.nan,np.nan,np.nan)
 p=r[r>0].sum();q=-r[r<0].sum();return n,float(r.mean()*1e4),float((r>0).mean()),float(p/max(q,1e-12))
D={}
for sym in SYMS:
 d=pd.read_csv(SRC/f'{sym}_3m_360d.csv');F=feat(d);X=F.to_numpy(float);valid=np.isfinite(X).all(1);valid[-45:]=False;rows=np.flatnonzero(valid);X=X[rows];r=barrier(d,rows);n=len(rows);hs=int(n*5/6);edges=np.linspace(0,hs,6,dtype=int);D[sym]=(X,r,hs,edges)
 print('DATA',sym,len(rows),X.shape[1],flush=True)
# pooled walk-forward, threshold on pooled validation, score on each coin test
foldcells=[]
for seg in [2,3,4]:
 Xtr=[];ytr=[];Xv=[];yv=[];Xt={};yt={}
 for sym,(X,r,hs,e) in D.items():
  tr=np.arange(0,e[seg-1]);va=np.arange(e[seg-1],e[seg]);te=np.arange(e[seg],e[seg+1]);Xtr.append(X[tr]);ytr.append(r[tr]);Xv.append(X[va]);yv.append(r[va]);Xt[sym]=X[te];yt[sym]=r[te]
 m=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,l2_regularization=5.,learning_rate=.05,random_state=9100+seg).fit(np.vstack(Xtr),np.concatenate(ytr));pv=m.predict(np.vstack(Xv))
 for fq in FRACS:
  th=float(np.quantile(pv,1-fq))
  for sym in SYMS:
   pt=m.predict(Xt[sym]);z=pt>=th;n,b,w,pf=met(yt[sym][z]);foldcells.append(dict(fold=seg,frac=fq,symbol=sym,n=n,bps=b,win=w,pf=pf))
R=pd.DataFrame(foldcells);R.to_csv(OUT/'SHORT_POOLED_3M_WF.csv',index=False)
choices=[]
for fq,g in R.groupby('frac'):
 gg=g[g.n>0];N=int(gg.n.sum());pos=int((gg.bps>0).sum());net=float(np.average(gg.bps,weights=np.maximum(gg.n,1)));worst=float(gg.bps.min());choices.append(dict(frac=fq,n=N,positive_cells=pos,net_bps=net,worst_bps=worst))
C=pd.DataFrame(choices).sort_values(['positive_cells','net_bps','n'],ascending=[False,False,False]);print('CHOICES');print(C.to_string(index=False),flush=True);E=C[(C.positive_cells>=4)&(C.n>=50)];ch=E.iloc[0] if len(E) else C.iloc[0];fq=float(ch.frac)
# final pooled model on both dev, threshold pooled dev, evaluate holdouts separately
Xdev=[];ydev=[]
for X,r,hs,e in D.values():Xdev.append(X[:hs]);ydev.append(r[:hs])
m=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,l2_regularization=5.,learning_rate=.05,random_state=9999).fit(np.vstack(Xdev),np.concatenate(ydev));pdv=m.predict(np.vstack(Xdev));th=float(np.quantile(pdv,1-fq));final=[]
for sym,(X,r,hs,e) in D.items():
 ph=m.predict(X[hs:]);z=ph>=th;n,b,w,pf=met(r[hs:][z]);res=dict(symbol=sym,frac=fq,wf_positive_cells=int(ch.positive_cells),wf_n=int(ch.n),wf_net_bps=float(ch.net_bps),wf_worst_bps=float(ch.worst_bps),threshold=th,hold_n=n,hold_bps=b,hold_win=w,hold_pf=pf);final.append(res);print('RESULT',json.dumps(res,sort_keys=True),flush=True)
pd.DataFrame(choices).to_csv(OUT/'SHORT_POOLED_3M_WF_SUMMARY.csv',index=False);pd.DataFrame(final).to_csv(OUT/'SHORT_POOLED_3M_HOLDOUT.csv',index=False);(OUT/'SHORT_POOLED_3M_HOLDOUT.json').write_text(json.dumps(final,indent=2))
