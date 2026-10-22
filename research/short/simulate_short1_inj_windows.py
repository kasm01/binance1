from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from models.short_model_registry import ShortModelRegistry
from features.short_economic import build_short_economic_features

DATA=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_360d.csv')
MODEL_DIR=Path('/home/kasm920/binance1/research/short/models')
df=pd.read_csv(DATA)
for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','open_time','close_time']:
    if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
if 'close_time' in df:
    df['ts']=pd.to_datetime(df['close_time'],unit='ms',utc=True,errors='coerce')
else:
    df['ts']=pd.to_datetime(df['open_time'],unit='ms',utc=True,errors='coerce')+pd.Timedelta(minutes=3)
f=build_short_economic_features(df)
m=ShortModelRegistry(MODEL_DIR).get('INJUSDT','3m',require_promoted=False)
valid=np.isfinite(f[m.features]).all(axis=1)
df['score']=np.nan
df.loc[valid,'score']=m.model.predict_proba(f.loc[valid,m.features])[:,1]
thr=float(m.meta['reference_probability_threshold_full_train'])

def sim(start, stop_roi, tp_roi=.0055, gb_roi=.0008, lev=2., fee=.0008, max_hold_bars=60):
    d=df[df.ts>=start].copy().reset_index(drop=True); trades=[]; i=0
    while i<len(d)-1:
        sc=d.loc[i,'score']
        if not np.isfinite(sc) or sc<thr: i+=1; continue
        ei=i+1; entry=float(d.loc[ei,'open']); best=0.; armed=False; xi=None; xp=None; reason='timeout'
        stop_px=entry*(1+abs(stop_roi)/lev); end=min(len(d),ei+max_hold_bars)
        for j in range(ei,end):
            hi=float(d.loc[j,'high']); lo=float(d.loc[j,'low'])
            if hi>=stop_px: xi=j; xp=stop_px; reason='hard_stop'; break
            if armed:
                trig=best-gb_roi; trig_px=entry*(1-trig/lev)
                if hi>=trig_px: xi=j; xp=trig_px; reason='tp_giveback'; break
            best=max(best,((entry-lo)/entry)*lev)
            if best>=tp_roi: armed=True
        if xi is None: xi=end-1; xp=float(d.loc[xi,'close'])
        gross=(entry-xp)/entry; net=gross-fee
        trades.append((d.loc[ei,'ts'],gross,net,reason,(xi-ei+1)*3))
        i=max(i+1,xi+1)
    t=pd.DataFrame(trades,columns=['ts','gross','net','reason','minutes'])
    if t.empty:return {'n':0}
    gp=t.loc[t.net>0,'net'].sum(); gl=-t.loc[t.net<0,'net'].sum(); eq=t.net.cumsum(); dd=(eq-eq.cummax()).min()
    return {'n':len(t),'net_sum_pct':t.net.sum()*100,'gross_sum_pct':t.gross.sum()*100,'win_pct':(t.net>0).mean()*100,'pf':gp/gl if gl>0 else math.inf,'max_dd_additive_pct':dd*100,'avg_net_bps':t.net.mean()*1e4,'hard_stop_n':int((t.reason=='hard_stop').sum()),'tp_giveback_n':int((t.reason=='tp_giveback').sum()),'timeout_n':int((t.reason=='timeout').sum()),'avg_minutes':t.minutes.mean()}

end=df.ts.max(); windows={'1M':end-pd.Timedelta(days=30),'6M':end-pd.Timedelta(days=182),'1Y':end-pd.Timedelta(days=365)}
print('PERIOD',df.ts.min(),df.ts.max(),'ROWS',len(df),'THRESHOLD',thr)
for stop in [-.006,-.03,-.06,-.135]:
    print('STOP_ROI',stop)
    for name,start in windows.items(): print(name,json.dumps(sim(start,stop),default=float))