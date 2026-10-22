#!/usr/bin/env python3
from pathlib import Path
import gzip,re,pandas as pd
OUT=Path('research/short/results')
OPEN_RE=re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*\[EXEC-SHORT\]\[OPENED\] symbol=([A-Z0-9]+) qty=([0-9.]+) price=([0-9.]+) notional=([0-9.]+) leverage=([0-9.]+)')
CLOSE_RE=re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*\[EXEC-SHORT\]\[CLOSED\] symbol=([A-Z0-9]+) reason=([^ ]+) qty=([0-9.]+) entry=([0-9.]+) exit=([0-9.]+) pnl_pct=([-0-9.]+) roi=([-0-9.]+) interval=([^ ]+)')
rows=[]
for p in Path('logs').rglob('*'):
 if not p.is_file() or not (p.suffix=='.gz' or p.name.startswith('main.log') or p.name.startswith('system.log')): continue
 try:
  op=gzip.open if p.suffix=='.gz' else open
  with op(p,'rt',encoding='utf-8',errors='ignore') as f:
   for line in f:
    if '[EXEC-SHORT][OPENED]' in line:
     m=OPEN_RE.search(line)
     if m:
      ts,s,q,px,n,l=m.groups();rows.append(dict(ts=pd.Timestamp(ts,tz='UTC'),type='OPEN',symbol=s,qty=float(q),price=float(px),notional=float(n),leverage=float(l),reason='',entry=float(px),exit=float('nan'),pnl_pct=float('nan'),roi=float('nan'),file=str(p)))
    elif '[EXEC-SHORT][CLOSED]' in line:
     m=CLOSE_RE.search(line)
     if m:
      ts,s,reason,q,en,ex,pnl,roi,itv=m.groups();rows.append(dict(ts=pd.Timestamp(ts,tz='UTC'),type='CLOSE',symbol=s,qty=float(q),price=float(ex),notional=float('nan'),leverage=float('nan'),reason=reason,entry=float(en),exit=float(ex),pnl_pct=float(pnl),roi=float(roi),file=str(p)))
 except Exception: pass
D=pd.DataFrame(rows)
if D.empty: raise SystemExit('no trade logs')
# exact event duplicates exist across rotated copies; retain one
key=['ts','type','symbol','qty','price','reason','entry']
D=D.sort_values('ts').drop_duplicates(key,keep='first').reset_index(drop=True)
D.to_csv(OUT/'SHORT_REAL_TRADE_EVENTS.csv',index=False)
active={};pairs=[];orph_open=0;orph_close=0
for _,r in D.iterrows():
 s=r.symbol
 if r.type=='OPEN':
  if s in active: orph_open+=1
  active[s]=r
 else:
  if s not in active:
   orph_close+=1; continue
  o=active.pop(s)
  # reject impossible pair where logged close entry materially differs from open fill
  rel=abs(float(r.entry)-float(o.price))/max(float(o.price),1e-12)
  pairs.append(dict(symbol=s,open_ts=o.ts,close_ts=r.ts,duration_sec=(r.ts-o.ts).total_seconds(),open_price=o.price,close_price=r.exit,qty=o.qty,notional=o.notional,leverage=o.leverage,reason=r.reason,pnl_pct=r.pnl_pct,roi=r.roi,entry_mismatch_pct=rel,open_file=o.file,close_file=r.file))
P=pd.DataFrame(pairs)
P=P[P.duration_sec>=0].copy()
P.to_csv(OUT/'SHORT_REAL_TRADES_PAIRED.csv',index=False)
print('EVENTS',len(D),'OPEN',int((D.type=='OPEN').sum()),'CLOSE',int((D.type=='CLOSE').sum()))
print('PAIRS',len(P),'orphan_open_overwrite',orph_open,'orphan_close',orph_close,'still_active',len(active))
if len(P):
 print('RANGE',P.open_ts.min(),P.close_ts.max())
 print('\nREASONS');print(P.groupby('reason').agg(n=('symbol','size'),mean_pnl=('pnl_pct','mean'),mean_roi=('roi','mean'),mean_duration_min=('duration_sec',lambda x:x.mean()/60)).to_string())
 print('\nSYMBOL');print(P.groupby('symbol').agg(n=('symbol','size'),mean_pnl=('pnl_pct','mean'),win=('pnl_pct',lambda x:(x>0).mean()),mean_roi=('roi','mean'),mean_duration_min=('duration_sec',lambda x:x.mean()/60)).to_string())
 print('\nALL',{'n':len(P),'mean_pnl_bps':P.pnl_pct.mean()*1e4,'win':float((P.pnl_pct>0).mean()),'mean_roi':P.roi.mean(),'mean_duration_min':P.duration_sec.mean()/60,'max_entry_mismatch_bps':P.entry_mismatch_pct.max()*1e4})
