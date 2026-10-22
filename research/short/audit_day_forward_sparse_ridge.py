#!/usr/bin/env python3
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path('/home/kasm920/binance1');sys.path.insert(0,str(ROOT))
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
SEL=['rv60','rv20','ret_6','ema100_slope3','ema100_rel','body_pct','ema10_slope3','ret_60','ema10_rel','ret_mean60','atr14_pct','down_frac20','rv10','ema50_slope3','imb_mean6','ret_3']
SRC=ROOT/'research/short/results/SHORT_CANDIDATE_LIFECYCLE_FEATURES.csv'; OUT=ROOT/'research/short/results/SHORT_DAY_FORWARD_RIDGE.csv'
FRAC=.20

def met(y):
 y=np.asarray(y,float);n=len(y)
 if not n:return (0,np.nan,np.nan,np.nan)
 p=y[y>0].sum();q=-y[y<0].sum();return n,float(y.mean()*1e4),float((y>0).mean()),float(p/max(q,1e-12))

def main():
 d=pd.read_csv(SRC);d['date']=pd.to_datetime(d.ts,utc=True).dt.date;dates=sorted(d.date.unique()); rows=[]
 # strict day-forward: at least first day train, each later day test independently
 for i in range(1,len(dates)):
  tr=d[d.date<dates[i]].copy(); te=d[d.date==dates[i]].copy()
  if len(tr)<100: continue
  sc=StandardScaler().fit(tr[SEL]);m=Ridge(alpha=10.0).fit(sc.transform(tr[SEL]),tr.net_return_pct)
  ptr=m.predict(sc.transform(tr[SEL]));pte=m.predict(sc.transform(te[SEL]));th=float(np.quantile(ptr,1-FRAC))
  base=te[te.state.eq('CONFIRM')]; sel=te[(pte>=th)&te.state.eq('CONFIRM')]
  bn,bb,bw,bpf=met(base.net_return_pct);sn,sb,sw,spf=met(sel.net_return_pct)
  rows.append(dict(test_date=str(dates[i]),train_n=len(tr),test_n=len(te),threshold=th,base_n=bn,base_bps=bb,base_pf=bpf,selected_n=sn,selected_bps=sb,selected_pf=spf))
 R=pd.DataFrame(rows);R.to_csv(OUT,index=False);print(R.to_string(index=False))
 valid=R[R.selected_n>0];
 if len(valid): print('WEIGHTED_SELECTED_BPS',np.average(valid.selected_bps,weights=valid.selected_n),'POS_DAYS',int((valid.selected_bps>0).sum()),'/',len(valid),'N',int(valid.selected_n.sum()))
if __name__=='__main__':main()
