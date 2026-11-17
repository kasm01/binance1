"""Observed closed 5m trend/flow at unique new SHORT READY times; shadow research."""
import csv,bisect,statistics
from pathlib import Path
path=Path('research/short/data/INJUSDT_5m_20261025_20261102_PENDING_REPLAY.csv')
m=list(csv.DictReader(path.open()))
rows=[dict(open_ms=int(x['open_ms']),close_ms=int(x['close_ms']),c=float(x['close']),o=float(x['open']),
   q=float(x['quote_volume']),b=float(x['taker_buy_quote'])) for x in m]
closes=[x['close_ms'] for x in rows]
pairs=list(csv.DictReader(open('research/results/SHORT1_INJ_PENDING_TO_FRESH_READY_ONE_TO_ONE_20261113.csv')))
out=[]
for item in pairs:
 i=bisect.bisect_left(closes,int(item['fresh_ms']))-1
 if i<12:continue
 past=rows[i-11:i+1]
 d5=[past[j]['c']/past[j-1]['c']-1 for j in range(9,12)]
 def flow(a):
  return sum(2*x['b']-x['q'] for x in a)/sum(x['q'] for x in a)
 f15=flow(past[-3:]);f60=flow(past)
 prior_trend=(statistics.mean(d5)<0 and past[-1]['c']<past[-2]['c'])
 aligned=prior_trend and f15<0 and f60<0
 item=dict(item,mean_last3_5m_pct=round(100*statistics.mean(d5),4),
    flow15=round(f15,4),flow60=round(f60,4),
    bearish_5m_dualflow=aligned)
 out.append(item)
destination=Path('research/results/SHORT1_INJ_NEW_READY_CLOSED_5M_CONTEXT_20261113.csv')
with destination.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]) if out else []);w.writeheader();w.writerows(out)
print('READY_COUNT',len(out),'BEARISH_5M_AND_NEGATIVE_15_60_FLOW',sum(x['bearish_5m_dualflow'] for x in out))
for kind in (True,False):
 a=[float(r['fresh_short_30m_pct']) for r in out if r['bearish_5m_dualflow']==kind]
 print('ALIGNED',kind,'N',len(a),'AVG_SHORT_30M_PRICE_PCT',round(statistics.mean(a),4) if a else None,
       'POSITIVE',sum(y>0 for y in a))
print('SAVED',destination,'LIMIT: no fresh 3m model source in old events, not an executable strategy')
