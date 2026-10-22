#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd,joblib,json
from sklearn.ensemble import HistGradientBoostingRegressor
PRE=Path('/home/kasm920/binance1-pro/data/offline_cache/LINKUSDT_1m_360d.csv')
FWD=Path('research/short/data/LINKUSDT_1m_FORWARD_20260914.csv')
OUT=Path('research/short/results/LINKUSDT_SHORT_TRUE_FORWARD.json')
TRADES=Path('research/short/results/LINKUSDT_SHORT_TRUE_FORWARD_TRADES.csv')
MODEL=Path('research/short/models/LINKUSDT_short_utility_regressor.joblib')
TP=.012;SL=.020;H=120;FEE=.0008;FRAC=.005

def build_features(d):
 x=d.copy();x['_g']=pd.to_numeric(x.open_time).astype('int64')//300000
 a=x.groupby('_g',sort=True).agg(open_time=('open_time','first'),close_time=('close_time','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),taker_buy_base_volume=('taker_buy_base_volume','sum')).reset_index(drop=True)
 o=a.open.astype(float);hi=a.high.astype(float);lo=a.low.astype(float);c=a.close.astype(float);v=a.volume.astype(float);tb=a.taker_buy_base_volume.astype(float);r1=c.pct_change();f=pd.DataFrame({'ts':a.close_time.astype('int64')})
 f['ret1']=r1;f['ret2']=c.pct_change(2);f['ret3']=c.pct_change(3);f['ret6']=c.pct_change(6);f['ret12']=c.pct_change(12);f['body']=(c-o)/o;f['range']=(hi-lo)/c;rng=(hi-lo).replace(0,np.nan);f['close_loc']=(c-lo)/rng;f['upper_wick']=(hi-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-lo)/c
 for span in [5,10,20,50]:
  e=c.ewm(span=span,adjust=False).mean();f[f'e{span}_rel']=c/e-1
  if span in [10,20,50]:f[f'e{span}_slope3']=e/e.shift(3)-1
 pc=c.shift(1);tr=pd.concat([hi-lo,(hi-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1);f['atr14']=tr.rolling(14).mean()/c
 for w in [10,20,60]:f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r1.rolling(w).std()
 buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
 for w in [3,6,12]:f[f'imb_mean{w}']=f.imbalance.rolling(w).mean();f[f'down_frac{w}']=(r1<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r1.rolling(w).mean()
 f['ret_accel']=r1-r1.shift(1);f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std();return f.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
def utility_for_entries(d, entry_idx):
 oo=d.open.to_numpy(float);hh=d.high.to_numpy(float);ll=d.low.to_numpy(float);cc=d.close.to_numpy(float);n=len(entry_idx);vals=[];state=[];exit_idx=[]
 CH=3000
 for st in range(0,n,CH):
  ids=entry_idx[st:st+CH];e=oo[ids];idx=ids[:,None]+np.arange(H)[None,:];adv=hh[idx]/e[:,None]-1;fav=1-ll[idx]/e[:,None];sh=adv>=SL;th=fav>=TP;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);ist=(ti<si)&(ti<H);timeout=1-cc[ids+H-1]/e;gross=np.where(isl,-SL,np.where(ist,TP,timeout));dur=np.where(isl,si+1,np.where(ist,ti+1,H));vals.append(gross-FEE);state.extend(np.where(isl,'SL',np.where(ist,'TP','TIMEOUT')).tolist());exit_idx.extend((ids+dur-1).tolist())
 return np.concatenate(vals),np.array(state),np.array(exit_idx)
# pre dataset
pre=pd.read_csv(PRE); fp=build_features(pre); ot=pre.open_time.to_numpy(np.int64); ei=np.searchsorted(ot,fp.ts.to_numpy(np.int64),side='right'); ok=ei+H<len(pre);fp=fp.loc[ok].reset_index(drop=True);ei=ei[ok];feat=[c for c in fp if c!='ts'];X=fp[feat].to_numpy('float32');y,_,_=utility_for_entries(pre,ei);n=len(fp);cut=int(n*5/6);tr=np.arange(cut);va=np.arange(cut,n)
rg=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=4001).fit(X[tr],y[tr]);ptr=rg.predict(X[tr]);pva=rg.predict(X[va]);mu=float(ptr.mean());sd=float(ptr.std()) or 1.;sva=(pva-mu)/sd;thr=float(np.quantile(sva,1-FRAC));mv=sva>=thr;val_net=float(y[va][mv].mean());val_n=int(mv.sum());health=bool(val_n>=50 and val_net>0);MODEL.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'model':rg,'features':feat,'mu':mu,'sd':sd,'score_thr':thr,'health':health,'val_net':val_net,'val_n':val_n,'tp':TP,'sl':SL,'horizon':H,'fee':FEE},MODEL)
# concatenate history tail + forward so rolling features are causal/continuous
fwd=pd.read_csv(FWD); combo=pd.concat([pre.tail(24*60*3),fwd],ignore_index=True).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True);fc=build_features(combo);cot=combo.open_time.to_numpy(np.int64);cei=np.searchsorted(cot,fc.ts.to_numpy(np.int64),side='right');fwd_start=int(fwd.open_time.iloc[0]);valid=(cei+H<len(combo))&(combo.open_time.iloc[np.minimum(cei,len(combo)-1)].to_numpy(np.int64)>=fwd_start);fc=fc.loc[valid].reset_index(drop=True);cei=cei[valid];XF=fc[feat].to_numpy('float32');sf=(rg.predict(XF)-mu)/sd;sel=(sf>=thr) if health else np.zeros(len(sf),bool);uy,st,ex=utility_for_entries(combo,cei);selected=np.flatnonzero(sel);vv=uy[sel];pos=vv[vv>0].sum();neg=-vv[vv<0].sum();entry_times=pd.to_datetime(combo.open_time.iloc[cei[selected]].to_numpy(np.int64),unit='ms',utc=True) if len(selected) else pd.DatetimeIndex([]);exit_times=pd.to_datetime(combo.close_time.iloc[ex[selected]].to_numpy(np.int64),unit='ms',utc=True) if len(selected) else pd.DatetimeIndex([])
td=pd.DataFrame({'entry_time':entry_times,'exit_time':exit_times,'score_z':sf[sel],'net_return':vv,'state':st[sel]});td.to_csv(TRADES,index=False)
res={'train_n':int(len(tr)),'validation_n':int(len(va)),'validation_selected':val_n,'validation_net_bps':val_net*1e4,'validation_win':float((y[va][mv]>0).mean()),'score_threshold_z':thr,'health_gate':health,'forward_rows':int(len(fc)),'forward_selected':int(sel.sum()),'forward_net_bps':(float(vv.mean()*1e4) if len(vv) else None),'forward_win':(float((vv>0).mean()) if len(vv) else None),'forward_pf':(float(pos/max(neg,1e-12)) if len(vv) else None),'forward_total_return_unscaled':(float(vv.sum()) if len(vv) else 0.0),'forward_tp':int((st[sel]=='TP').sum()),'forward_sl':int((st[sel]=='SL').sum()),'forward_timeout':int((st[sel]=='TIMEOUT').sum()),'feature_count':len(feat)}
OUT.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2));print('MODEL',MODEL,'TRADES',TRADES)
