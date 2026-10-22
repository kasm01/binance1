#!/usr/bin/env python3
from pathlib import Path
import gzip, re
import numpy as np
import pandas as pd

P = pd.read_csv('research/short/results/SHORT_REAL_TRADES_PAIRED.csv')
P['open_ts'] = pd.to_datetime(P.open_ts, utc=True)
P = P.sort_values('open_ts')
pat = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*\[EXEC-SHORT\]\[DECISION\] symbol=([A-Z0-9]+) score=([0-9.]+).*final_min=([0-9.]+)')
rows = []
for p in Path('logs').rglob('*'):
    if not p.is_file() or not (p.suffix == '.gz' or p.name.startswith('main.log') or p.name.startswith('system.log')):
        continue
    try:
        op = gzip.open if p.suffix == '.gz' else open
        with op(p, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '[EXEC-SHORT][DECISION]' not in line:
                    continue
                m = pat.search(line)
                if m:
                    ts, s, sc, th = m.groups()
                    rows.append((pd.Timestamp(ts, tz='UTC'), s, float(sc), float(th)))
    except Exception:
        pass
D = pd.DataFrame(rows, columns=['decision_ts','symbol','model_score','threshold']).drop_duplicates().sort_values('decision_ts')
out = []
for _, r in P.iterrows():
    g = D[(D.symbol == r.symbol) & (D.decision_ts <= r.open_ts) & (D.decision_ts >= r.open_ts - pd.Timedelta(seconds=5))]
    z = g.iloc[-1] if len(g) else None
    q = r.to_dict()
    q['model_score'] = float(z.model_score) if z is not None else np.nan
    q['decision_lag_sec'] = (r.open_ts - z.decision_ts).total_seconds() if z is not None else np.nan
    out.append(q)
J = pd.DataFrame(out)
J['net8'] = J.pnl_pct - 0.0008
J.to_csv('research/short/results/SHORT_REAL_TRADES_WITH_MODEL_SCORE.csv', index=False)
M = J.dropna(subset=['model_score']).copy()
print('MATCHED', len(M), '/', len(J), 'lag_median', M.decision_lag_sec.median())
print('score range', M.model_score.min(), M.model_score.max(), 'spearman', M[['model_score','net8']].corr(method='spearman').iloc[0,1])
M['q'] = pd.qcut(M.model_score, 4, duplicates='drop')
print('\nSCORE QUARTILES @8bps')
for q, g in M.groupby('q', observed=True):
    r = g.net8
    pos = r[r > 0].sum(); neg = -r[r < 0].sum()
    print(q, 'n', len(g), 'score_mean', round(g.model_score.mean(),5), 'net_bps', round(r.mean()*1e4,3), 'pf', round(float(pos/max(neg,1e-12)),3), 'win', round(float((r>0).mean()),3))
print('\nSCORE >= CUTS @8bps')
for c in [.500,.501,.502,.503,.504,.505,.507,.510,.515,.520]:
    g = M[M.model_score >= c]
    if len(g) < 15:
        continue
    r = g.net8
    pos = r[r > 0].sum(); neg = -r[r < 0].sum()
    print(c, 'n', len(g), 'net', round(r.mean()*1e4,3), 'pf', round(float(pos/max(neg,1e-12)),3))
