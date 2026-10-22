#!/usr/bin/env python3
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SYMS=['LINKUSDT','AVAXUSDT'];SRC=Path('/home/kasm920/binance1-pro/data/offline_cache');OUT=Path('research/short/results');OUT.mkdir(parents=True,exist_ok=True)
FEE=.0008;TP=.006;SL=.010;H=40;FRACS=[.0025,.005,.01,.02,.05]
REGIMES=['all','ret240_neg','ret480_neg','ema100_below','ema200_below','ret480_ema200','ret240_ema100','trend_stack']

def build(d):
 o=d.open.astype(float);h=d.high.astype(float);l=d.low.astype(float);c=d.close.astype(float);v=d.volume.astype(float);tb=d.taker_buy_base_volume.astype(float);r=c.pct_change();f=pd.DataFrame(index=d.index)
 for n in [1,2,3,6,12,20,60,240,480]:f[f'ret{n}']=c.pct_change(n)
 f['body']=(c-o)/o;f['range']=(h-l)/c;rng=(h-l).replace(0,np.nan);f['close_loc']=(c-l)/rng;f['upper_wick']=(h-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-l)/c
 for span in [5,10,20,50,100,200]:
  e=c.ewm(span=span,adjust=False).mean();f[f'e{span}_rel']=c/e-1;f[f'e{span}_slope3']=e/e.shift(3)-1
 pc=c.shift(1);tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1);f['atr14']=tr.rolling(14).mean()/c
 for w in [10,20,60,240]:f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r.rolling(w).std();f[f'down_frac{w}']=(r<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r.rolling(w).mean()
 buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
 for w in [3,6,12,30]:f[f'imb_mean{w}']=f.imbalance.rolling(w).mean()
 f['ret_accel']=r-r.shift(1);f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std()
 return f.replace([np.inf,-np.inf],np.nan)

def regime_mask(F,name):
 if name=='all':return np.ones(len(F),bool)
 if name=='ret240_neg':return F.ret240.to_numpy()<0
 if name=='ret480_neg':return F.ret480.to_numpy()<0
 if name=='ema100_below':return F.e100_rel.to_numpy()<0
 if name=='ema200_below':return F.e200_rel.to_numpy()<0
 if name=='ret480_ema200':return (F.ret480.to_numpy()<0)&(F.e200_rel.to_numpy()<0)
 if name=='ret240_ema100':return (F.ret240.to_numpy()<0)&(F.e100_rel.to_numpy()<0)
 if name=='trend_stack':return (F.ret480.to_numpy()<0)&(F.e100_rel.to_numpy()<0)&(F.e200_rel.to_numpy()<0)
 raise KeyError(name)

def barrier(d,rows):
 o=d.open.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float);cl=d.close.to_numpy(float);e=o[rows+1];idx=(rows+1)[:,None]+np.arange(H)[None,:];sh=(hi[idx]/e[:,None]-1)>=SL;th=(1-lo[idx]/e[:,None])>=TP;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);itp=(ti<si)&(ti<H);tout=1-cl[rows+H]/e-FEE;return np.where(isl,-SL-FEE,np.where(itp,TP-FEE,tout))
def met(r):
 r=np.asarray(r);n=len(r)
 if not n:return (0,np.nan,np.nan,np.nan)
 p=r[r>0].sum();q=-r[r<0].sum();return n,float(r.mean()*1e4),float((r>0).mean()),float(p/max(q,1e-12))

final=[];foldrec=[]
for sym in SYMS:
 d=pd.read_csv(SRC/f'{sym}_3m_360d.csv');F=build(d);cols=list(F.columns);X=F.to_numpy(float);valid=np.isfinite(X).all(1);valid[-45:]=False;rows=np.flatnonzero(valid);X=X[rows];Fc=F.iloc[rows].reset_index(drop=True);r=barrier(d,rows);n=len(rows);hs=int(n*5/6);dev=np.arange(hs);hold=np.arange(hs,n);edges=np.linspace(0,hs,6,dtype=int)
 # fit each fold once, save predictions
 preds=[]
 for seg in [2,3,4]:
  tr=np.arange(0,edges[seg-1]);va=np.arange(edges[seg-1],edges[seg]);te=np.arange(edges[seg],edges[seg+1]);m=HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=15,l2_regularization=4.,learning_rate=.05,random_state=8000+seg).fit(X[tr],r[tr]);preds.append((seg,va,te,m.predict(X[va]),m.predict(X[te])))
 choices=[]
 for reg in REGIMES:
  rm=regime_mask(Fc,reg)
  for fq in FRACS:
   fs=[]
   for seg,va,te,pv,pt in preds:
    # threshold learned only among validation rows in same regime
    vv=pv[rm[va]]
    if len(vv)<40:continue
    th=float(np.quantile(vv,1-fq));z=rm[te]&(pt>=th);n0,b,w,pf=met(r[te][z]);fs.append((seg,n0,b,w,pf))
   if not fs:continue
   G=pd.DataFrame(fs,columns=['fold','n','bps','win','pf']);N=int(G.n.sum());pos=int((G.bps>0).sum());net=float(np.average(G.bps,weights=np.maximum(G.n,1)));worst=float(G.bps.min());choices.append(dict(regime=reg,frac=fq,n=N,posfold=pos,net=net,worst=worst))
 C=pd.DataFrame(choices).sort_values(['posfold','net','n'],ascending=[False,False,False]);E=C[(C.posfold>=2)&(C.n>=30)];ch=E.iloc[0] if len(E) else C.iloc[0];reg=str(ch.regime);fq=float(ch.frac)
 # lock and holdout
 m=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,l2_regularization=4.,learning_rate=.05,random_state=9999).fit(X[dev],r[dev]);pdv=m.predict(X[dev]);ph=m.predict(X[hold]);rm=regime_mask(Fc,reg);vv=pdv[rm[dev]];th=float(np.quantile(vv,1-fq));z=rm[hold]&(ph>=th);n0,b,w,pf=met(r[hold][z]);res=dict(symbol=sym,regime=reg,frac=fq,wf_posfold=int(ch.posfold),wf_n=int(ch.n),wf_bps=float(ch.net),wf_worst=float(ch.worst),threshold=th,hold_n=n0,hold_bps=b,hold_win=w,hold_pf=pf);final.append(res);print('RESULT',json.dumps(res,sort_keys=True),flush=True)
 # save choices for audit
 C.insert(0,'symbol',sym);foldrec.extend(C.to_dict('records'))
pd.DataFrame(foldrec).to_csv(OUT/'SHORT_REGIME_CONTEXT_3M_WF_CHOICES.csv',index=False);pd.DataFrame(final).to_csv(OUT/'SHORT_REGIME_CONTEXT_3M_HOLDOUT.csv',index=False);(OUT/'SHORT_REGIME_CONTEXT_3M_HOLDOUT.json').write_text(json.dumps(final,indent=2));print('\nFINAL');print(pd.DataFrame(final).to_string(index=False))
