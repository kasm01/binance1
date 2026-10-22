#!/usr/bin/env python3
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
SEL=['rv60','rv20','ret_6','ema100_slope3','ema100_rel','body_pct','ema10_slope3','ret_60','ema10_rel','ret_mean60','atr14_pct','down_frac20','rv10','ema50_slope3','imb_mean6','ret_3']
SRC='research/short/results/SHORT_CANDIDATE_LIFECYCLE_FEATURES.csv'; FRACS=[.1,.2,.3,.5]

def fit(X,y): return Pipeline([('sc',StandardScaler()),('m',LogisticRegression(C=.25,max_iter=1000,class_weight='balanced',random_state=1))]).fit(X,y)
def met(x):
 x=np.asarray(x,float);n=len(x)
 if not n:return (0,np.nan,np.nan,np.nan)
 p=x[x>0].sum();q=-x[x<0].sum();return n,float(x.mean()*1e4),float((x>0).mean()),float(p/max(q,1e-12))
def main():
 d=pd.read_csv(SRC).sort_values('ts').reset_index(drop=True);X=d[SEL].to_numpy(float);ret=d.net_return_pct.to_numpy(float);y=(ret>0).astype(int);conf=d.state.eq('CONFIRM').to_numpy();n=len(d);hs=int(n*.8);dev=np.arange(hs);hold=np.arange(hs,n);edges=np.linspace(0,hs,6,dtype=int)
 rec=[]
 for fq in FRACS:
  fs=[]
  for seg in [2,3,4]:
   tr=np.arange(0,edges[seg-1]);va=np.arange(edges[seg-1],edges[seg]);te=np.arange(edges[seg],edges[seg+1]);m=fit(X[tr],y[tr]);pv=m.predict_proba(X[va])[:,1];pt=m.predict_proba(X[te])[:,1];th=float(np.quantile(pv,1-fq));z=(pt>=th)&conf[te];fs.append(met(ret[te][z]))
  N=sum(x[0] for x in fs);pos=sum(x[1]>0 for x in fs if x[0]);bps=float(np.average([x[1] for x in fs if x[0]],weights=[x[0] for x in fs if x[0]])) if N else np.nan;worst=min([x[1] for x in fs if x[0]],default=np.nan);rec.append((fq,N,pos,bps,worst))
 R=pd.DataFrame(rec,columns=['frac','wf_n','wf_posfold','wf_bps','wf_worst']).sort_values(['wf_posfold','wf_worst','wf_bps'],ascending=[False,False,False]);print('WF\n',R.to_string(index=False));E=R[(R.wf_posfold>=2)&(R.wf_n>=15)];ch=E.iloc[0] if len(E) else R.iloc[0];fq=float(ch.frac)
 m=fit(X[dev],y[dev]);pdv=m.predict_proba(X[dev])[:,1];ph=m.predict_proba(X[hold])[:,1];th=float(np.quantile(pdv,1-fq));z=(ph>=th)&conf[hold];base=met(ret[hold][conf[hold]]);sel=met(ret[hold][z]);print('SELECT frac',fq,'prob_threshold',th);print('HOLD_BASE',base);print('HOLD_SELECTED',sel)
 print('COEF')
 co=m.named_steps['m'].coef_[0]; order=np.argsort(np.abs(co))[::-1]
 for i in order: print(SEL[i],round(float(co[i]),4))
if __name__=='__main__':main()
