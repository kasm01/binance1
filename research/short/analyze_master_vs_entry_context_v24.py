#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.context.market_context_contract import classify_confirmed_position_build

ONE = ROOT / 'research/short/data/inj_1m'
SCORES = ROOT / 'research/short/results/SHORT1_INJ_WF_SCORES_V4.csv'
OI = ROOT / 'research/short/data/INJUSDT_metrics_5m_20251012_20261013.csv'
ENTRIES = ROOT / 'research/short/results/SHORT1_OI_PREVBAR_V23/EXH20_RISKMULT50_CONFIRMED_BUILD/entries.csv'
OUT = ROOT / 'research/short/results/SHORT1_MASTER_VS_ENTRY_CONTEXT_V24'
OUT.mkdir(parents=True, exist_ok=True)
BASE_TH = 0.540


def load_context() -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted(ONE.glob('*.csv'))]
    d = pd.concat(parts, ignore_index=True)
    d['ts'] = pd.to_datetime(d.open_time, unit='ms', utc=True)
    for c in ['close','quote_volume','taker_buy_quote_volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    sell = d.quote_volume - d.taker_buy_quote_volume
    net = d.taker_buy_quote_volume - sell
    d['flow15'] = net.rolling(15,min_periods=15).sum() / d.quote_volume.rolling(15,min_periods=15).sum()
    d['ret15'] = d.close.pct_change(15)
    d['flow15_completed'] = d.flow15.shift(1)
    d['ret15_completed'] = d.ret15.shift(1)

    m = pd.read_csv(OI)
    m['oi_ts'] = pd.to_datetime(m.create_time, utc=True)
    m = m.sort_values('oi_ts').reset_index(drop=True)
    m['oi15'] = pd.to_numeric(m.sum_open_interest, errors='coerce').pct_change(3)
    d['oi_lookup_ts'] = d.ts - pd.Timedelta(nanoseconds=1)
    d = pd.merge_asof(
        d.sort_values('oi_lookup_ts'),
        m[['oi_ts','oi15']],
        left_on='oi_lookup_ts', right_on='oi_ts',
        direction='backward', tolerance=pd.Timedelta('5m'),
    )
    return d.sort_values('ts').reset_index(drop=True)


def classify(row: pd.Series) -> dict:
    return classify_confirmed_position_build(
        oi_delta_15m=float(row.oi15) if np.isfinite(row.oi15) else 0.0,
        price_return_15m=float(row.ret15_completed) if np.isfinite(row.ret15_completed) else 0.0,
        money_flow_15m=float(row.flow15_completed) if np.isfinite(row.flow15_completed) else 0.0,
    )


def main() -> None:
    d = load_context()
    d = d.set_index('ts', drop=False)
    s = pd.read_csv(SCORES)
    s['score_ts'] = pd.to_datetime(s.ts, utc=True)
    s['eligible_ts'] = s.score_ts + pd.Timedelta(minutes=3)
    s['wf_score'] = pd.to_numeric(s.wf_score, errors='coerce')
    s = s[np.isfinite(s.wf_score) & (s.wf_score >= BASE_TH)].sort_values('eligible_ts')
    score_map = {t: float(v) for t,v in zip(s.eligible_ts,s.wf_score)}

    e = pd.read_csv(ENTRIES)
    e['entry_ts'] = pd.to_datetime(e.ts, utc=True)
    e['confirmation_ts'] = e.entry_ts - pd.Timedelta(minutes=1)

    rows=[]
    missing=0
    for rec in e.itertuples(index=False):
        confirm = rec.confirmation_ts
        born = None
        # Candidate is valid in born minute and the immediately following minute.
        for t in (confirm, confirm-pd.Timedelta(minutes=1)):
            if t in score_map:
                # Prefer exact score match when available.
                if abs(score_map[t] - float(rec.score)) < 1e-8:
                    born=t; break
                if born is None:
                    born=t
        if born is None or born not in d.index or confirm not in d.index:
            missing += 1
            continue
        mr=classify(d.loc[born]); er=classify(d.loc[confirm])
        mn,md=mr['name'],mr['build_direction']; en,ed=er['name'],er['build_direction']
        if mn==en and (mn!='confirmed_build' or md==ed):
            transition='same_confirmed_build' if mn=='confirmed_build' else 'same_normal'
        elif mn=='normal' and en=='confirmed_build': transition='normal_to_confirmed_build'
        elif mn=='confirmed_build' and en=='normal': transition='confirmed_build_to_normal'
        elif mn=='confirmed_build' and en=='confirmed_build' and md!=ed: transition='confirmed_build_direction_flip'
        else: transition=f'{mn}_to_{en}'
        master_mult=.5 if mn=='confirmed_build' else 1.0
        entry_mult=.5 if en=='confirmed_build' else 1.0
        rows.append({
            'entry_ts':rec.entry_ts,'confirmation_ts':confirm,'master_ts':born,
            'lag_min':int((confirm-born)/pd.Timedelta(minutes=1)),
            'score':float(rec.score),
            'master_regime':mn,'master_direction':md,
            'entry_regime':en,'entry_direction':ed,
            'transition':transition,
            'sizing_signal_changed':master_mult!=entry_mult,
            'master_context_multiplier':master_mult,
            'entry_context_multiplier':entry_mult,
            'master_oi15':mr['oi_delta_15m'],'entry_oi15':er['oi_delta_15m'],
            'master_ret15':mr['price_return_15m'],'entry_ret15':er['price_return_15m'],
            'master_flow15':mr['money_flow_15m'],'entry_flow15':er['money_flow_15m'],
        })
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'comparisons.csv',index=False)
    n=len(out)
    print('ENTRIES',len(e),'MATCHED',n,'MISSING',missing)
    if n==0: return
    c=Counter(out.transition)
    print('TRANSITIONS')
    for k,v in c.most_common(): print(f'{k:34s} {v:5d} {100*v/n:7.3f}%')
    changed=int((out.transition.str.startswith('same_')==False).sum())
    sizing=int(out.sizing_signal_changed.sum())
    print('CHANGED',changed,f'{100*changed/n:.3f}%')
    print('SIZING_SIGNAL_CHANGED',sizing,f'{100*sizing/n:.3f}%')
    print('MASTER_BUILD',int((out.master_regime=='confirmed_build').sum()),f"{100*(out.master_regime=='confirmed_build').mean():.3f}%")
    print('ENTRY_BUILD',int((out.entry_regime=='confirmed_build').sum()),f"{100*(out.entry_regime=='confirmed_build').mean():.3f}%")
    print('LAG')
    for k,v in Counter(out.lag_min).items(): print(k,v,f'{100*v/n:.3f}%')
    pd.DataFrame([{'transition':k,'count':v,'pct':100*v/n} for k,v in c.most_common()]).to_csv(OUT/'transition_summary.csv',index=False)

if __name__=='__main__': main()
