from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import joblib
from features.short_economic import build_short_economic_features

DATA=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_360d.csv')
MODEL_PATH=Path('research/short/models/short_model_INJUSDT_3m_allbar_v3.joblib')
META_PATH=Path('research/short/models/short_model_meta_INJUSDT_3m_allbar_v3.json')
FEE=0.0008; LEV=3.0; TP_ARM=.0400; GIVEBACK=.0024
SL_PRICE=.0299; CATA_PRICE=.035; MAX_LAYERS=3; COOLDOWN_BARS=1
DD_REDUCE=.08; DD_RECOVER=.04
NORMAL=(1.00,.50,2.00); REDUCED=(.50,.10,.70)
START_EQUITY=1000.0

def load():
    d=pd.read_csv(DATA)
    for c in ['open','high','low','close','open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['ts']=pd.to_datetime(d['open_time'],unit='ms',utc=True)
    f=build_short_economic_features(d)
    meta=json.loads(META_PATH.read_text())
    model=joblib.load(MODEL_PATH); feats=list(meta['features'])
    valid=np.isfinite(f[feats]).all(axis=1)
    d['score']=np.nan; d.loc[valid,'score']=model.predict_proba(f.loc[valid,feats])[:,1]
    for c in ['ret_60','ret_240','atr14_pct','rv60','ema100_rel','ema100_slope3','down_frac20','imb_mean6']:
        d[c]=f[c]
    class M: pass
    m=M(); m.meta=meta
    return d,m

def replay(d,start,threshold,gate_name='NONE'):
    x=d[d.ts>=start].copy().reset_index(drop=True)
    eq=START_EQUITY; peak=eq; mode='NORMAL'; position=None; last_entry_i=-10**9
    closed=[]; entries=[]; curve=[]; reduce_count=0; normal_count=0
    i=0
    while i<len(x)-1:
        row=x.iloc[i]; hi=float(row.high); lo=float(row.low); cl=float(row.close)
        if position is not None:
            ent=position['avg_entry']; qty=position['qty']; cata_px=ent*(1+CATA_PRICE); soft_px=ent*(1+SL_PRICE)
            exit_px=None; reason=None
            if hi>=cata_px:
                exit_px=cata_px; reason='catastrophic'
            else:
                breached=hi>=soft_px
                if breached and position.get('soft_pending',False): exit_px=soft_px; reason='soft_stop_confirmed'
                position['soft_pending']=bool(breached)
                if not breached: position['soft_pending']=False
                if exit_px is None and position.get('armed',False):
                    trig=position['best_roi']-GIVEBACK; trig_px=ent*(1-trig/LEV)
                    if hi>=trig_px: exit_px=trig_px; reason='tp_giveback'
                if exit_px is None:
                    fav_roi=((ent-lo)/ent)*LEV
                    if fav_roi>position['best_roi']: position['best_roi']=fav_roi
                    if position['best_roi']>=TP_ARM: position['armed']=True
            if exit_px is not None:
                gross=(ent-exit_px)*qty; exit_fee=exit_px*qty*(FEE/2); pnl=gross-exit_fee
                eq+=pnl; closed.append({'ts':row.ts,'pnl':pnl,'reason':reason,'layers':position['layers'],'eq':eq,'dur_bars':i-position['open_i']+1}); position=None
                peak=max(peak,eq)
        peak=max(peak,eq); dd=(eq/peak-1) if peak>0 else 0
        if mode=='NORMAL' and dd<=-DD_REDUCE: mode='REDUCED'; reduce_count+=1
        elif mode=='REDUCED' and dd>=-DD_RECOVER: mode='NORMAL'; normal_count+=1
        sc=float(row.score) if np.isfinite(row.score) else float('nan')
        gate=True
        if gate_name=='RET240_ATR_RV': gate=(float(row.ret_240)<=0 and float(row.atr14_pct)<=ATR90 and float(row.rv60)<=RV90)
        elif gate_name=='RET240_ATR': gate=(float(row.ret_240)<=0 and float(row.atr14_pct)<=ATR90)
        elif gate_name=='RET60_RET240_ATR': gate=(float(row.ret_60)<=0 and float(row.ret_240)<=0 and float(row.atr14_pct)<=ATR90)
        elif gate_name=='EMA100_ATR_RV': gate=(float(row.ema100_rel)<=0 and float(row.atr14_pct)<=ATR90 and float(row.rv60)<=RV90)
        can_signal=np.isfinite(sc) and sc>=threshold and gate and (i-last_entry_i)>=COOLDOWN_BARS
        layers=0 if position is None else position['layers']
        if can_signal and layers<MAX_LAYERS:
            ei=i+1; px=float(x.iloc[ei].open)
            layer1,add,cap=(NORMAL if mode=='NORMAL' else REDUCED)
            pct=layer1 if layers==0 else add
            target_notional=eq*pct; current_notional=0 if position is None else position['notional']; max_notional=eq*cap
            notional=min(target_notional,max(0.0,max_notional-current_notional))
            if notional>=10 and px>0:
                qty=notional/px; eq-=notional*(FEE/2)
                if position is None:
                    position={'qty':qty,'avg_entry':px,'notional':notional,'layers':1,'best_roi':0.0,'armed':False,'soft_pending':False,'open_i':ei}
                else:
                    nq=position['qty']+qty; position['avg_entry']=(position['avg_entry']*position['qty']+px*qty)/nq
                    position['qty']=nq; position['notional']+=notional; position['layers']+=1
                    position['best_roi']=0.0; position['armed']=False; position['soft_pending']=False
                entries.append({'ts':x.iloc[ei].ts,'score':sc,'layer':position['layers'],'notional':notional,'mode':mode,'eq':eq,'price':px})
                last_entry_i=i
        mtm=eq
        if position is not None: mtm+=(position['avg_entry']-cl)*position['qty']
        curve.append((row.ts,eq,mtm,mode,0 if position is None else position['layers']))
        i+=1
    open_pos=None
    if position is not None:
        px=float(x.iloc[-1].close); gross=(position['avg_entry']-px)*position['qty']; fee=px*position['qty']*(FEE/2); pnl=gross-fee
        open_pos={'unrealized_net':pnl,'layers':position['layers'],'avg_entry':position['avg_entry'],'last_price':px}
    c=pd.DataFrame(closed); e=pd.DataFrame(entries); q=pd.DataFrame(curve,columns=['ts','equity','mtm','mode','layers'])
    realized_ret=(eq/START_EQUITY-1)*100; total_ret=((eq+(open_pos['unrealized_net'] if open_pos else 0))/START_EQUITY-1)*100
    if len(c):
        pos=c.loc[c.pnl>0,'pnl'].sum(); neg=-c.loc[c.pnl<0,'pnl'].sum(); pf=pos/neg if neg>0 else math.inf; win=(c.pnl>0).mean()*100
    else: pf=float('nan'); win=float('nan')
    mtm_peak=q.mtm.cummax(); maxdd=((q.mtm/mtm_peak)-1).min()*100 if len(q) else 0
    return {'start':str(start),'end':str(x.ts.max()),'threshold':threshold,'entries':len(e),'closed_positions':len(c),'realized_return_pct':realized_ret,'total_return_with_open_pct':total_ret,'final_realized_equity':eq,'win_pct':win,'pf':pf,'max_mtm_dd_pct':maxdd,'layer1':int((e.layer==1).sum()) if len(e) else 0,'layer2':int((e.layer==2).sum()) if len(e) else 0,'layer3':int((e.layer==3).sum()) if len(e) else 0,'reduced_entries':int((e['mode']=='REDUCED').sum()) if len(e) else 0,'normal_entries':int((e['mode']=='NORMAL').sum()) if len(e) else 0,'dd_reduce_transitions':reduce_count,'dd_recover_transitions':normal_count,'open_position':open_pos,'reasons':c.reason.value_counts().to_dict() if len(c) else {}}

d,m=load(); thr=float(m.meta['reference_probability_threshold_full_train']); end=d.ts.max()
train_cut=int(len(d)*.70); ATR90=float(d.atr14_pct.iloc[:train_cut].quantile(.90)); RV90=float(d.rv60.iloc[:train_cut].quantile(.90))
print('DATA',d.ts.min(),end,'ROWS',len(d),'THRESHOLD',thr,'ATR90',ATR90,'RV90',RV90)
for gate in ['RET240_ATR','RET60_RET240_ATR']:
    print('GATE',gate)
    for name,days in [('1M',30),('6M',182),('1Y',365)]:
        print(name,json.dumps(replay(d,end-pd.Timedelta(days=days),thr,gate),default=float,sort_keys=True))
