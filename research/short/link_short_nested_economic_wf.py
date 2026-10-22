#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier

SRC=Path('/home/kasm920/binance1-pro/data/offline_cache/LINKUSDT_1m_360d.csv')
OUT=Path('/home/kasm920/binance1/research/short/results/LINKUSDT_SHORT_NESTED_ECON_WF.csv')
SUM=Path('/home/kasm920/binance1/research/short/results/LINKUSDT_SHORT_NESTED_ECON_WF_SUMMARY.csv')
FEE=.0008
CONFIGS=[(.006,.020,120),(.006,.015,120),(.006,.010,120),(.008,.020,120),(.012,.020,120)]
BLENDS=[0.,.25,.5,.75,1.]
FRACS=[.005,.01,.02,.05,.10]

d=pd.read_csv(SRC)
# completed 5m bars
x=d.copy(); x['_g']=pd.to_numeric(x.open_time).astype('int64')//300000
a=x.groupby('_g',sort=True).agg(open_time=('open_time','first'),close_time=('close_time','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),quote_asset_volume=('quote_asset_volume','sum'),number_of_trades=('number_of_trades','sum'),taker_buy_base_volume=('taker_buy_base_volume','sum'),taker_buy_quote_volume=('taker_buy_quote_volume','sum')).reset_index(drop=True)
o=a.open.astype(float);h=a.high.astype(float);l=a.low.astype(float);c=a.close.astype(float);v=a.volume.astype(float);tb=a.taker_buy_base_volume.astype(float)
r1=c.pct_change(); f=pd.DataFrame({'ts':a.close_time.astype('int64')})
f['ret1']=r1; f['ret2']=c.pct_change(2); f['ret3']=c.pct_change(3); f['ret6']=c.pct_change(6); f['ret12']=c.pct_change(12)
f['body']=(c-o)/o; f['range']=(h-l)/c; rng=(h-l).replace(0,np.nan); f['close_loc']=(c-l)/rng; f['upper_wick']=(h-np.maximum(o,c))/c; f['lower_wick']=(np.minimum(o,c)-l)/c
for span in [5,10,20,50]:
 e=c.ewm(span=span,adjust=False).mean(); f[f'e{span}_rel']=c/e-1
 if span in [10,20,50]: f[f'e{span}_slope3']=e/e.shift(3)-1
pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); f['atr14']=tr.rolling(14).mean()/c
for w in [10,20,60]:
 f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std(); f[f'rv{w}']=r1.rolling(w).std()
buy=tb/v.replace(0,np.nan); f['buy_ratio']=buy; f['imbalance']=2*buy-1
for w in [3,6,12]:
 f[f'imb_mean{w}']=f.imbalance.rolling(w).mean(); f[f'down_frac{w}']=(r1<0).astype(float).rolling(w).mean(); f[f'ret_mean{w}']=r1.rolling(w).mean()
f['ret_accel']=r1-r1.shift(1); f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std()
f=f.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)

# entry starts strictly after completed 5m bar
one_ot=pd.to_numeric(d.open_time).to_numpy(np.int64); oo=d.open.to_numpy(float); hh=d.high.to_numpy(float); ll=d.low.to_numpy(float); cc=d.close.to_numpy(float)
entry_idx=np.searchsorted(one_ot,f.ts.to_numpy(np.int64),side='right')
valid_entry=entry_idx+max(H for _,_,H in CONFIGS)<len(d)
f=f.loc[valid_entry].reset_index(drop=True); entry_idx=entry_idx[valid_entry]
feat=[col for col in f.columns if col!='ts']; X=f[feat].to_numpy('float32'); n=len(f)
print('DATASET',n,'features',len(feat),flush=True)

# utilities vectorized per config
utilities={}; states={}
CH=3000
for tp,sl,H in CONFIGS:
 vals=[]; wins=[]
 for st in range(0,n,CH):
  ids=entry_idx[st:st+CH]; e=oo[ids]; idx=ids[:,None]+np.arange(H)[None,:]
  adv=hh[idx]/e[:,None]-1.; fav=1.-ll[idx]/e[:,None]
  slh=adv>=sl; tph=fav>=tp
  sli=np.where(slh.any(1),slh.argmax(1),H+1); tpi=np.where(tph.any(1),tph.argmax(1),H+1)
  is_sl=(sli<=tpi)&(sli<=H-1); is_tp=(tpi<sli)&(tpi<=H-1)
  timeout=1.-cc[ids+H-1]/e
  gross=np.where(is_sl,-sl,np.where(is_tp,tp,timeout)); vals.append(gross-FEE); wins.append(gross>FEE)
 utilities[(tp,sl,H)]=np.concatenate(vals); states[(tp,sl,H)]=np.concatenate(wins)

# 6 chronological segments => outer tests segments 3,4,5; val immediately preceding
edges=np.linspace(0,n,7,dtype=int)
rows=[]
for cfg in CONFIGS:
 y=utilities[cfg]; win=states[cfg]
 for test_seg in [3,4,5]:
  tr=np.arange(edges[0],edges[test_seg-1]); va=np.arange(edges[test_seg-1],edges[test_seg]); te=np.arange(edges[test_seg],edges[test_seg+1])
  kw=dict(max_iter=90,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=2000+test_seg)
  rg=HistGradientBoostingRegressor(**kw).fit(X[tr],y[tr])
  cl=HistGradientBoostingClassifier(**kw).fit(X[tr],(y[tr]>0).astype(int))
  pr_tr=rg.predict(X[tr]); pp_tr=cl.predict_proba(X[tr])[:,1]
  pr_va=rg.predict(X[va]); pp_va=cl.predict_proba(X[va])[:,1]
  pr_te=rg.predict(X[te]); pp_te=cl.predict_proba(X[te])[:,1]
  def z(v,ref):
   sd=float(np.std(ref)); return (v-float(np.mean(ref)))/(sd if sd>1e-9 else 1.)
  best=None
  for blend in BLENDS:
   sv=blend*z(pr_va,pr_tr)+(1-blend)*z(pp_va,pp_tr)
   for frac in FRACS:
    thr=float(np.quantile(sv,1-frac)); sel=sv>=thr
    if sel.sum()<50: continue
    net=float(y[va][sel].mean()); obj=net*np.sqrt(sel.sum())
    cand=(obj,net,blend,frac,thr,int(sel.sum()),float((y[va][sel]>0).mean()))
    if best is None or cand[0]>best[0]: best=cand
  if best is None: continue
  _,vnet,blend,frac,thr,vn,vwin=best
  ste=blend*z(pr_te,pr_tr)+(1-blend)*z(pp_te,pp_tr); sel=ste>=thr
  vv=y[te][sel]
  pos=vv[vv>0].sum(); neg=-vv[vv<0].sum()
  rows.append(dict(tp=cfg[0],sl=cfg[1],horizon=cfg[2],fold=test_seg,train_n=len(tr),val_n=len(va),test_n=len(te),blend=blend,top_frac=frac,score_thr=thr,val_selected=vn,val_net_bps=vnet*1e4,val_win=vwin,test_selected=int(sel.sum()),test_net_bps=(float(vv.mean())*1e4 if len(vv) else np.nan),test_win=(float((vv>0).mean()) if len(vv) else np.nan),test_pf=(float(pos/max(neg,1e-12)) if len(vv) else np.nan),base_test_bps=float(y[te].mean()*1e4)))
R=pd.DataFrame(rows); R.to_csv(OUT,index=False)
agg=[]
for cfg,g in R.groupby(['tp','sl','horizon']):
 w=g.test_selected.clip(lower=1); valid=g.test_selected>0
 agg.append(dict(tp=cfg[0],sl=cfg[1],horizon=cfg[2],folds=len(g),selected=int(g.test_selected.sum()),net_bps=float(np.average(g.loc[valid,'test_net_bps'],weights=g.loc[valid,'test_selected'])) if valid.any() else np.nan,positive_folds=int((g.test_net_bps>0).sum()),worst_fold_bps=float(g.test_net_bps.min()),win=float(np.average(g.loc[valid,'test_win'],weights=g.loc[valid,'test_selected'])) if valid.any() else np.nan,pf_mean=float(g.test_pf.mean()),base_bps=float(g.base_test_bps.mean()),val_net_bps=float(g.val_net_bps.mean()),fracs=','.join(map(str,g.top_frac.tolist())),blends=','.join(map(str,g.blend.tolist()))))
A=pd.DataFrame(agg).sort_values('net_bps',ascending=False); A.to_csv(SUM,index=False)
print('\nFOLDS'); print(R.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nSUMMARY'); print(A.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('OUT',OUT,SUM)
