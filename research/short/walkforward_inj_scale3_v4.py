from pathlib import Path
import json, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, build_short_economic_features

DATA=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
H=20; FAV=.006; ADV=.0075; TEST_DAYS=30; INIT_TRAIN_DAYS=90
FRACS=(.05,.10,.15,.20); BAD_CTX_UPLIFT=.03
FEE=.0008; LEV=3.0; TP_ARM=.04; GIVEBACK=.0024; SL_PRICE=.0299; CATA_PRICE=.035
MAX_LAYERS=3; COOLDOWN_BARS=1; DD_REDUCE=.08; DD_RECOVER=.04
NORMAL=(1.0,.50,2.0); REDUCED=(.50,.10,.70); START_EQ=1000.0

def make_model(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])

d=pd.read_csv(DATA)
for c in ['open','high','low','close','open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True); f=build_short_economic_features(d)
o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float)
rows=np.arange(0,len(d)-H-1,dtype=int); ei=rows+1; fut=ei[:,None]+np.arange(H)[None,:]; ent=o[ei]
mfe=np.max(1-lo[fut]/ent[:,None],axis=1); mae=np.max(hi[fut]/ent[:,None]-1,axis=1); y=((mfe>=FAV)&(mae<=ADV)).astype(int)
X=f.iloc[rows].copy(); good=np.isfinite(X[SHORT_ECON_MODEL_FEATURES_V1].to_numpy(float)).all(axis=1)
rows=rows[good]; X=X.loc[good].reset_index(drop=True); y=y[good]; ts=d.ts.iloc[rows].reset_index(drop=True)
score=np.full(len(d),np.nan); base_thr=np.full(len(d),np.nan); atr90=np.full(len(d),np.nan); fold_id=np.full(len(d),-1,dtype=int)
start_time=ts.iloc[0]+pd.Timedelta(days=INIT_TRAIN_DAYS); end_time=ts.iloc[-1]; fold=0; fold_rows=[]
while start_time < end_time:
    test_end=min(start_time+pd.Timedelta(days=TEST_DAYS),end_time+pd.Timedelta(minutes=1))
    test_mask=(ts>=start_time)&(ts<test_end); test_idx=np.flatnonzero(test_mask.to_numpy())
    if not len(test_idx): start_time=test_end; continue
    safe_train_end=start_time-pd.Timedelta(minutes=66); train_idx=np.flatnonzero((ts<safe_train_end).to_numpy())
    if len(train_idx)<10000: start_time=test_end; continue
    split=max(1000,int(len(train_idx)*.80)); inner=train_idx[:split]; val=train_idx[split:]
    m0=make_model().fit(X.iloc[inner][SHORT_ECON_MODEL_FEATURES_V1],y[inner]); ptr=m0.predict_proba(X.iloc[inner][SHORT_ECON_MODEL_FEATURES_V1])[:,1]; pval=m0.predict_proba(X.iloc[val][SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    choices=[]
    for frac in FRACS:
        th=float(np.quantile(ptr,1-frac)); sel=pval>=th; n=int(sel.sum()); hit=float(y[val][sel].mean()) if n else -1.0; choices.append((hit,n,frac))
    eligible=[z for z in choices if z[1]>=500]; _,_,frac=max(eligible,key=lambda z:(z[0],z[1])) if eligible else max(choices,key=lambda z:z[1])
    m=make_model().fit(X.iloc[train_idx][SHORT_ECON_MODEL_FEATURES_V1],y[train_idx]); ptrain=m.predict_proba(X.iloc[train_idx][SHORT_ECON_MODEL_FEATURES_V1])[:,1]; th=float(np.quantile(ptrain,1-frac)); ptest=m.predict_proba(X.iloc[test_idx][SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    orig=rows[test_idx]; score[orig]=ptest; base_thr[orig]=th; fold_id[orig]=fold; a90=float(X.iloc[train_idx].atr14_pct.quantile(.90)); atr90[orig]=a90
    fold_rows.append({'fold':fold,'test_start':str(start_time),'test_end':str(test_end),'train_n':len(train_idx),'test_n':len(test_idx),'frac':frac,'threshold':th,'atr90':a90,'test_base_rate':float(y[test_idx].mean()),'test_selected_n':int((ptest>=th).sum()),'test_selected_hit':float(y[test_idx][ptest>=th].mean()) if (ptest>=th).any() else None})
    fold+=1; start_time=test_end

d['wf_score']=score; d['wf_threshold']=base_thr; d['wf_atr90']=atr90; d['wf_fold']=fold_id
for c in ['ret_240','atr14_pct']: d[c]=f[c]
first_oos=d.loc[np.isfinite(d.wf_score),'ts'].min(); last_oos=d.loc[np.isfinite(d.wf_score),'ts'].max()

def replay(start,policy):
    x=d[(d.ts>=start)&np.isfinite(d.wf_score)].copy().reset_index(drop=True)
    eq=START_EQ; peak=eq; mode='NORMAL'; pos=None; last_entry=-10**9; entries=[]; closed=[]; curve=[]
    for i in range(len(x)-1):
        r=x.iloc[i]; hi0=float(r.high); lo0=float(r.low); cl=float(r.close)
        if pos is not None:
            ent0=pos['avg']; cata=ent0*(1+CATA_PRICE); soft=ent0*(1+SL_PRICE); exit_px=None; reason=None
            if hi0>=cata: exit_px=cata; reason='catastrophic'
            else:
                breached=hi0>=soft
                if breached and pos.get('pending',False): exit_px=soft; reason='soft_stop_confirmed'
                pos['pending']=bool(breached)
                if not breached: pos['pending']=False
                if exit_px is None and pos['armed']:
                    trig=pos['best']-GIVEBACK; tpx=ent0*(1-trig/LEV)
                    if hi0>=tpx: exit_px=tpx; reason='tp_giveback'
                if exit_px is None:
                    roi=((ent0-lo0)/ent0)*LEV; pos['best']=max(pos['best'],roi); pos['armed']=pos['best']>=TP_ARM
            if exit_px is not None:
                pnl=(ent0-exit_px)*pos['qty']-exit_px*pos['qty']*(FEE/2); eq+=pnl; closed.append((r.ts,pnl,reason,pos['layers'])); pos=None; peak=max(peak,eq)
        peak=max(peak,eq); dd=eq/peak-1 if peak>0 else 0
        if mode=='NORMAL' and dd<=-DD_REDUCE: mode='REDUCED'
        elif mode=='REDUCED' and dd>=-DD_RECOVER: mode='NORMAL'
        sc=float(r.wf_score); th=float(r.wf_threshold); goodctx=(float(r.ret_240)<=0 and float(r.atr14_pct)<=float(r.wf_atr90)); eff=th+(BAD_CTX_UPLIFT if policy=='THRESHOLD' and not goodctx else 0.0)
        layers=0 if pos is None else pos['layers']; can=sc>=eff and (i-last_entry)>=COOLDOWN_BARS
        if can and layers<MAX_LAYERS:
            px=float(x.iloc[i+1].open); cfg=NORMAL if mode=='NORMAL' else REDUCED; pct=cfg[0] if layers==0 else cfg[1]; cur=0 if pos is None else pos['notional']; notional=min(eq*pct,max(0,eq*cfg[2]-cur))
            if notional>=10:
                q=notional/px; eq-=notional*(FEE/2)
                if pos is None: pos={'qty':q,'avg':px,'notional':notional,'layers':1,'best':0.,'armed':False,'pending':False}
                else:
                    nq=pos['qty']+q; pos['avg']=(pos['avg']*pos['qty']+px*q)/nq; pos['qty']=nq; pos['notional']+=notional; pos['layers']+=1; pos['best']=0.; pos['armed']=False; pos['pending']=False
                entries.append((x.iloc[i+1].ts,pos['layers'],notional,mode,goodctx,sc,th)); last_entry=i
        mtm=eq+(0 if pos is None else (pos['avg']-cl)*pos['qty']); curve.append(mtm)
    unreal=0.0
    if pos is not None:
        px=float(x.iloc[-1].close); unreal=(pos['avg']-px)*pos['qty']-px*pos['qty']*(FEE/2)
    c=pd.DataFrame(closed,columns=['ts','pnl','reason','layers']); e=pd.DataFrame(entries,columns=['ts','layer','notional','mode','context_good','score','threshold']); q=np.asarray(curve,float); qpk=np.maximum.accumulate(q) if len(q) else np.array([START_EQ]); mdd=float(np.min(q/qpk-1)*100) if len(q) else 0
    posp=float(c.loc[c.pnl>0,'pnl'].sum()) if len(c) else 0; neg=float(-c.loc[c.pnl<0,'pnl'].sum()) if len(c) else 0
    return {'start':str(start),'end':str(x.ts.max()),'policy':policy,'entries':len(e),'closed':len(c),'realized_return_pct':(eq/START_EQ-1)*100,'total_return_pct':((eq+unreal)/START_EQ-1)*100,'win_pct':float((c.pnl>0).mean()*100) if len(c) else None,'pf':posp/neg if neg>0 else None,'max_mtm_dd_pct':mdd,'normal_entries':int((e['mode']=='NORMAL').sum()) if len(e) else 0,'reduced_entries':int((e['mode']=='REDUCED').sum()) if len(e) else 0,'good_context_entries':int(e.context_good.sum()) if len(e) else 0,'bad_context_entries':int((~e.context_good).sum()) if len(e) else 0,'reasons':c.reason.value_counts().to_dict() if len(c) else {}}

print('OOS',first_oos,last_oos,'FOLDS',len(fold_rows)); print('FOLDS_JSON',json.dumps(fold_rows,default=float))
for policy in ['BASE','THRESHOLD']:
    print('POLICY',policy)
    for name,days in [('1M',30),('6M',182),('1Y',365)]: print(name,json.dumps(replay(last_oos-pd.Timedelta(days=days),policy),default=float,sort_keys=True))
# true forward unseen by prior Sep14 cache
for policy in ['BASE','THRESHOLD']:
    print('TRUE_FORWARD',policy,json.dumps(replay(pd.Timestamp('2026-09-14T20:24:00Z'),policy),default=float,sort_keys=True))
Path('research/short/results/SHORT1_INJ_WALKFORWARD_V4.json').write_text(json.dumps({'oos_start':str(first_oos),'oos_end':str(last_oos),'folds':fold_rows,'base_1y':replay(last_oos-pd.Timedelta(days=365),'BASE'),'threshold_1y':replay(last_oos-pd.Timedelta(days=365),'THRESHOLD'),'true_forward_base':replay(pd.Timestamp('2026-09-14T20:24:00Z'),'BASE'),'true_forward_threshold':replay(pd.Timestamp('2026-09-14T20:24:00Z'),'THRESHOLD')},indent=2,allow_nan=False,default=float))
