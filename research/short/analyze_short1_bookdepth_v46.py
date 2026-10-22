#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd, numpy as np

ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_BOOKDEPTH_V45B/SHORT1_BOOKDEPTH_V45B.csv'
OUT=ROOT/'research/short/results/SHORT1_BOOKDEPTH_V46'
OUT.mkdir(parents=True,exist_ok=True)
SPLIT=pd.Timestamp('2026-04-13',tz='UTC')

def prep():
 d=pd.read_csv(SRC)
 d['entry_ts']=pd.to_datetime(d.entry_ts,utc=True)
 num=[c for c in d.columns if c.startswith('bd') or c.startswith('flow_') or c.startswith('ret_') or c in ['mae5','mfe5','score']]
 for c in num:d[c]=pd.to_numeric(d[c],errors='coerce')
 d['adverse']=d.touch40_5m.eq('ADVERSE_FIRST');d['favorable']=d.touch40_5m.eq('FAVORABLE_FIRST')
 d['actionable20']=d.ret_20s<=-0.0001
 d['buy_flow_price_down']=d.actionable20 & (d.flow_20s>=0)
 d['sell_flow_price_down']=d.actionable20 & (d.flow_20s<0)
 d['risk20']=d.actionable20 & (d.flow_20s<=-0.60) & (d.flow_delta_5_20<=-0.30)
 d['rebound20_30']=d.ret_30s>d.ret_20s
 d['era']=np.where(d.entry_ts<SPLIT,'H1','H2')
 for band in ['1p0','0p2']:
  d[f'book_bull_{band}']=d[f'bd_{band}_notional_imb_delta_0_30']>0
  d[f'book30_bid_{band}']=d[f'bd30_{band}_notional_imb']>0
  d[f'bull_count_{band}']=(d[f'bd_{band}_notional_imb_delta_0_30']>0).astype(int)+(d[f'bd_{band}_bid_notional_ratio_0_30']>1).astype(int)+(d[f'bd_{band}_ask_notional_ratio_0_30']<1).astype(int)
 return d

def stats(z):
 if len(z)==0:return {'n':0}
 return {'n':int(len(z)),'adverse_first_rate':float(z.adverse.mean()),'favorable_first_rate':float(z.favorable.mean()),
         'mae5_median':float(z.mae5.median()),'mfe5_median':float(z.mfe5.median()),
         'touch':z.touch40_5m.value_counts().to_dict()}

def main():
 d=prep(); report={'rows':len(d),'split':str(SPLIT),'eras':{},'overall':{}}
 # fixed, pre-specified sign interactions; no threshold fitting here.
 for era in ['ALL','H1','H2']:
  x=d if era=='ALL' else d[d.era==era]
  r={}
  for band in ['1p0','0p2']:
   valid=x[f'bd30_{band}_notional_imb'].notna()
   r[f'{band}_actionable']=stats(x[x.actionable20 & valid])
   bp=x.buy_flow_price_down & valid
   r[f'{band}_buyflow_pricedown_bookbull']=stats(x[bp & x[f'book_bull_{band}']])
   r[f'{band}_buyflow_pricedown_bookbear']=stats(x[bp & ~x[f'book_bull_{band}']])
   rr=x.risk20 & x.rebound20_30 & valid
   r[f'{band}_risk_rebound_bookbull']=stats(x[rr & x[f'book_bull_{band}']])
   r[f'{band}_risk_rebound_bookbear']=stats(x[rr & ~x[f'book_bull_{band}']])
   # component-count view
   for k in [0,1,2,3]:
    r[f'{band}_buyflow_pricedown_bullcount{k}']=stats(x[bp & (x[f'bull_count_{band}']==k)])
  report['eras'][era]=r
 # save compact event table for exact inspection
 keep=['entry_ts','score','touch40_5m','mae5','mfe5','flow_20s','flow_delta_5_20','ret_20s','ret_30s','risk20','rebound20_30','era']
 for b in ['1p0','0p2']:
  keep += [f'bd30_{b}_notional_imb',f'bd_{b}_notional_imb_delta_0_30',f'bd_{b}_bid_notional_ratio_0_30',f'bd_{b}_ask_notional_ratio_0_30',f'bull_count_{b}']
 d[keep].to_csv(OUT/'events.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
 print('SAVED',OUT)
if __name__=='__main__':main()
