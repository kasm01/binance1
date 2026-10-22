#!/usr/bin/env python3
from pathlib import Path
import io,zipfile,requests
import numpy as np,pandas as pd
ROOT=Path('/home/kasm920/binance1'); SRC=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V37.csv'; CACHE=ROOT/'research/short/data/aggtrades_risk60_v40'; CACHE.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/data/futures/um/daily/aggTrades/INJUSDT'; cuts=[30,35,40,45,50,55,59.9]
S=requests.Session();S.headers.update({'User-Agent':'binance1-research-v40'})
def load_day(day,mins):
 cp=CACHE/f'{day}.csv.gz'
 if cp.exists():return pd.read_csv(cp)
 old=ROOT/f'research/short/data/aggtrades_small/INJUSDT-aggTrades-{day}.zip'
 if old.exists(): blob=old.read_bytes()
 else:
  r=S.get(f'{BASE}/INJUSDT-aggTrades-{day}.zip',timeout=30);r.raise_for_status();blob=r.content
 with zipfile.ZipFile(io.BytesIO(blob)) as z:
  with z.open(z.namelist()[0]) as f:d=pd.read_csv(f)
 d.columns=[c.strip().lower() for c in d.columns];t=pd.to_numeric(d.transact_time,errors='coerce').astype('Int64');
 if float(t.dropna().median())>1e14:t=t//1000
 d['transact_time_ms']=t;mm=(t//60000)*60000;z=d[mm.isin(set(mins))][['price','quantity','first_trade_id','last_trade_id','transact_time_ms','is_buyer_maker']].copy();z.to_csv(cp,index=False,compression='gzip');return z
def maker(s):return s.astype(str).str.lower().isin(['true','1','t']).to_numpy(bool)
def feat(z,ms,op,sec):
 x=z[pd.to_numeric(z.transact_time_ms)<=ms+int(sec*1000)]
 if x.empty:return np.nan,np.nan
 p=pd.to_numeric(x.price).to_numpy(float);q=pd.to_numeric(x.quantity).to_numpy(float);qq=p*q;sg=np.where(maker(x.is_buyer_maker),-qq,qq)
 return float(p[-1]/op-1),float(sg.sum()/qq.sum())
def stat(x):
 p=x.pnl_net.dropna().to_numpy(float);gp=p[p>0].sum();gl=-p[p<0].sum();return {'n':len(x),'adv':x.adv.mean(),'fav':x.fav.mean(),'win':x.final_win.mean(),'pnl':x.pnl_net.sum(),'pf':gp/gl if gl else np.nan,'mae':x.mae5.median(),'mfe':x.mfe5.median()}
def main():
 d=pd.read_csv(SRC);d['confirm_ts']=pd.to_datetime(d.confirm_ts,utc=True);d['adv']=d.touch40_5m.eq('ADVERSE_FIRST');d['fav']=d.touch40_5m.eq('FAVORABLE_FIRST');d['risk']=(d.flow_20s<=-.6)&(d.flow_delta_5_20<=-.3);r=d[d.risk].copy();print('risk',len(r),'days',r.confirm_ts.dt.date.nunique())
 rows=[]
 for day,g in r.groupby(r.confirm_ts.dt.strftime('%Y-%m-%d')):
  mins=[int(x.value//1_000_000) for x in g.confirm_ts];z=load_day(day,mins)
  for x in g.itertuples():
   ms=int(x.confirm_ts.value//1_000_000);zz=z[((pd.to_numeric(z.transact_time_ms)//60000)*60000)==ms];o={}
   for sec in cuts:o[f'ret_{sec:g}s'],o[f'flow_{sec:g}s']=feat(zz,ms,float(x.confirm_open),sec)
   rows.append((x.Index if hasattr(x,'Index') else None,o))
 # index-safe rebuild in current order
 extra=[]
 for x in r.itertuples(index=False):
  day=x.confirm_ts.strftime('%Y-%m-%d');z=pd.read_csv(CACHE/f'{day}.csv.gz');ms=int(x.confirm_ts.value//1_000_000);zz=z[((pd.to_numeric(z.transact_time_ms)//60000)*60000)==ms];o={}
  for sec in cuts:o[f'ret_{sec:g}s'],o[f'flow_{sec:g}s']=feat(zz,ms,float(x.confirm_open),sec)
  extra.append(o)
 R=pd.concat([r.reset_index(drop=True),pd.DataFrame(extra)],axis=1);R.to_csv(ROOT/'research/short/results/SHORT1_PENDING_REVALIDATION_V40.csv',index=False)
 base_rebound=R.ret_delta_20_30>=0
 print('BASE RISK',stat(R));print('30_CONT',stat(R[~base_rebound]));print('30_REBOUND',stat(R[base_rebound]))
 for sec in [35,40,45,50,55,59.9]:
  col=f'ret_{sec:g}s'; resume=base_rebound & (R[col]<R.ret_20s); still=base_rebound & ~resume
  print(f'\nAT {sec}s among 30s rebound:')
  print('RESUMED_BELOW_RET20',stat(R[resume]));print('STILL_NOT_BELOW_RET20',stat(R[still]))
 print('cache_kb',round(sum(p.stat().st_size for p in CACHE.glob('*.csv.gz'))/1024,1))
if __name__=='__main__':main()
