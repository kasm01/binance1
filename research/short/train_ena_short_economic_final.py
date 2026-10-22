#!/usr/bin/env python3
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SYMBOL='ENAUSDT'; TP=.008; SL=.020; H=24; FEE=.0008; TOP_FRAC=.01
SRC=Path(f'research/short/data/candidate_5m_360d/{SYMBOL}_5m_360d.csv')
OUT=Path('research/short/models'); OUT.mkdir(parents=True,exist_ok=True)

def build(d):
 o=d.open.astype(float);h=d.high.astype(float);l=d.low.astype(float);c=d.close.astype(float);v=d.volume.astype(float);tb=d.taker_buy_base_volume.astype(float);r1=c.pct_change();f=pd.DataFrame({'ts':d.close_time.astype('int64')})
 f['ret1']=r1;f['ret2']=c.pct_change(2);f['ret3']=c.pct_change(3);f['ret6']=c.pct_change(6);f['ret12']=c.pct_change(12);f['body']=(c-o)/o;f['range']=(h-l)/c;rng=(h-l).replace(0,np.nan);f['close_loc']=(c-l)/rng;f['upper_wick']=(h-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-l)/c
 for span in [5,10,20,50]:
  e=c.ewm(span=span,adjust=False).mean();f[f'e{span}_rel']=c/e-1
  if span in [10,20,50]:f[f'e{span}_slope3']=e/e.shift(3)-1
 pc=c.shift(1);tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1);f['atr14']=tr.rolling(14).mean()/c
 for w in [10,20,60]:f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r1.rolling(w).std()
 buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
 for w in [3,6,12]:f[f'imb_mean{w}']=f.imbalance.rolling(w).mean();f[f'down_frac{w}']=(r1<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r1.rolling(w).mean()
 f['ret_accel']=r1-r1.shift(1);f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std()
 return f.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
def utility(d,rows):
 o=d.open.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float);cl=d.close.to_numpy(float);out=[]
 for st in range(0,len(rows),4000):
  ids=rows[st:st+4000];e=o[ids];idx=ids[:,None]+np.arange(H)[None,:];adv=hi[idx]/e[:,None]-1;fav=1-lo[idx]/e[:,None];sh=adv>=SL;th=fav>=TP;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);ist=(ti<si)&(ti<H);timeout=1-cl[ids+H-1]/e;gross=np.where(isl,-SL,np.where(ist,TP,timeout));out.append(gross-FEE)
 return np.concatenate(out)

d=pd.read_csv(SRC); f=build(d); ot=d.open_time.to_numpy(np.int64); ei=np.searchsorted(ot,f.ts.to_numpy(np.int64),side='right'); ok=ei+H<len(d);f=f.loc[ok].reset_index(drop=True);ei=ei[ok];features=[c for c in f if c!='ts'];X=f[features].to_numpy('float32');y=utility(d,ei)
# causal 24h regime at completed setup bar
c=d.close.astype(float);mom24=c.pct_change(288).to_numpy();sigidx=np.maximum(ei-1,0);regime=np.nan_to_num(mom24[sigidx],nan=1.0)<0
model=HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=8111).fit(X,y)
pred=model.predict(X);mu=float(pred.mean());sd=float(pred.std()) or 1.;z=(pred-mu)/sd;thr=float(np.quantile(z[regime],1-TOP_FRAC))
sel=regime & (z>=thr);ys=y[sel];pos=ys[ys>0].sum();neg=-ys[ys<0].sum()
model_path=OUT/f'{SYMBOL}_short_economic_5m.joblib';meta_path=OUT/f'{SYMBOL}_short_economic_5m.json';joblib.dump(model,model_path)
meta=dict(symbol=SYMBOL,interval='5m',model_type='HistGradientBoostingRegressor',target='short_net_utility',tp_price_pct=TP,sl_price_pct=SL,horizon_bars=H,horizon_minutes=H*5,roundtrip_fee_rate=FEE,market_context='mom24h<0',market_context_bars=288,top_frac=TOP_FRAC,pred_mean=mu,pred_std=sd,score_threshold_z=thr,features=features,n_samples=len(y),selected=int(sel.sum()),selected_net_bps=float(ys.mean()*1e4),selected_win=float((ys>0).mean()),selected_pf=float(pos/max(neg,1e-12)),training_start=str(pd.to_datetime(d.open_time.iloc[0],unit='ms',utc=True)),training_end=str(pd.to_datetime(d.close_time.iloc[-1],unit='ms',utc=True)),research_note='Final full-data fit after nested/holdout research; future shadow validation still required.')
meta_path.write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2));print(model_path,meta_path)
