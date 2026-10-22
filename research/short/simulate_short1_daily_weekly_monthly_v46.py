#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np, pandas as pd

ROOT=Path('/home/kasm920/binance1')
BASE=ROOT/'research/short/results/SHORT1_COMBINED_V30/C600'
MICRO=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V37.csv'
OUT=ROOT/'research/short/results/SHORT1_SIM_V46'; OUT.mkdir(parents=True,exist_ok=True)
START=1000.0
FLOW20_MAX=-0.60
DELTA_MAX=-0.30
REV=0.0001

def pf(x):
    x=pd.to_numeric(x,errors='coerce').dropna().to_numpy(float)
    gp=x[x>0].sum(); gl=-x[x<0].sum()
    return float(gp/gl) if gl>0 else None

def maxdd(series):
    a=pd.to_numeric(series,errors='coerce').dropna().to_numpy(float)
    if not len(a):return None
    pk=np.maximum.accumulate(a)
    return float(np.min(a/pk-1)*100)

def period_table(curve,closed,freq,label):
    q=curve.copy(); q['ts']=pd.to_datetime(q.ts,utc=True)
    if freq=='D': q['period']=q.ts.dt.strftime('%Y-%m-%d')
    elif freq=='W':
        iso=q.ts.dt.isocalendar(); q['period']=iso.year.astype(str)+'-W'+iso.week.astype(str).str.zfill(2)
    elif freq=='M': q['period']=q.ts.dt.strftime('%Y-%m')
    rows=[]
    c=closed.copy(); c['close_ts']=pd.to_datetime(c.close_ts,utc=True)
    for p,g in q.groupby('period',sort=True):
        start_eq=float(g.iloc[0].mtm); end_eq=float(g.iloc[-1].mtm)
        if freq=='D': mask=c.close_ts.dt.strftime('%Y-%m-%d').eq(p)
        elif freq=='W':
            ci=c.close_ts.dt.isocalendar(); cp=ci.year.astype(str)+'-W'+ci.week.astype(str).str.zfill(2); mask=cp.eq(p)
        else: mask=c.close_ts.dt.strftime('%Y-%m').eq(p)
        cc=c[mask]
        rows.append({'period':p,'return_pct':(end_eq/start_eq-1)*100 if start_eq else np.nan,'closed':len(cc),'pnl_net':float(cc.pnl_net.sum()) if len(cc) else 0.0,'pf':pf(cc.pnl_net),'period_maxdd_pct':maxdd(g.mtm)})
    z=pd.DataFrame(rows); z.to_csv(OUT/f'{label}.csv',index=False); return z

def stat(z):
    r=z.return_pct.dropna()
    return {'periods':len(z),'positive_rate':float((r>0).mean()) if len(r) else None,'median_return_pct':float(r.median()) if len(r) else None,
            'mean_return_pct':float(r.mean()) if len(r) else None,'p05_return_pct':float(r.quantile(.05)) if len(r) else None,
            'worst_return_pct':float(r.min()) if len(r) else None,'best_return_pct':float(r.max()) if len(r) else None,
            'median_closed':float(z.closed.median()) if len(z) else None,'worst_period':str(z.loc[z.return_pct.idxmin(),'period']) if len(r) else None,
            'best_period':str(z.loc[z.return_pct.idxmax(),'period']) if len(r) else None}

def rolling(curve,days):
    q=curve.copy(); q['ts']=pd.to_datetime(q.ts,utc=True); q=q.set_index('ts').sort_index()
    # use daily end MTM for robust calendar rolling windows
    d=q.mtm.resample('1D').last().dropna()
    r=(d/d.shift(days)-1)*100
    return r.dropna()

def hazard_overlay():
    d=pd.read_csv(MICRO); d['entry_ts']=pd.to_datetime(d.entry_ts,utc=True)
    for c in ['flow_5s','flow_20s','ret_20s','ret_30s','pnl_net','mae5','mfe5','score']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    # V37 rows are frozen baseline C600 entries; previous-green confirmation already held by construction.
    # Here we only test whether the live-style reversal threshold had already been reached by 20s.
    confirmed20=((-d.ret_20s).clip(lower=0)>=REV)
    d['hazard_pending']=confirmed20 & (d.flow_20s<=FLOW20_MAX) & ((d.flow_20s-d.flow_5s)<=DELTA_MAX)
    d['hazard_reject_candidate']=d.hazard_pending & ((d.ret_30s-d.ret_20s)>=0)
    d['hazard_ready_after_revalidation']=d.hazard_pending & ((d.ret_30s-d.ret_20s)<0)
    d['touch_adverse']=d.touch40_5m.eq('ADVERSE_FIRST')
    d['touch_favorable']=d.touch40_5m.eq('FAVORABLE_FIRST')
    def summ(g):
        return {'n':int(len(g)),'win_rate':float(pd.to_numeric(g.final_win,errors='coerce').mean()) if len(g) else None,'pnl_diag_sum':float(g.pnl_net.sum()) if len(g) else 0.0,
                'pf_diag':pf(g.pnl_net),'adverse_first_rate':float(g.touch_adverse.mean()) if len(g) else None,'favorable_first_rate':float(g.touch_favorable.mean()) if len(g) else None,
                'mae5_median':float(g.mae5.median()) if len(g) else None,'mfe5_median':float(g.mfe5.median()) if len(g) else None}
    out={'all':summ(d),'pending':summ(d[d.hazard_pending]),'reject_candidate':summ(d[d.hazard_reject_candidate]),'ready_after_revalidation':summ(d[d.hazard_ready_after_revalidation])}
    # calendar concentration diagnostic
    d['day']=d.entry_ts.dt.strftime('%Y-%m-%d'); iso=d.entry_ts.dt.isocalendar(); d['week']=iso.year.astype(str)+'-W'+iso.week.astype(str).str.zfill(2); d['month']=d.entry_ts.dt.strftime('%Y-%m')
    for key in ['day','week','month']:
        g=d[d.hazard_reject_candidate].groupby(key).agg(reject_n=('entry_ts','size'),reject_pnl_diag=('pnl_net','sum'),adverse_rate=('touch_adverse','mean')).reset_index()
        g.to_csv(OUT/f'hazard_reject_by_{key}.csv',index=False)
    d[['entry_ts','score','pnl_net','final_win','touch40_5m','hazard_pending','hazard_reject_candidate','hazard_ready_after_revalidation','flow_5s','flow_20s','ret_20s','ret_30s','mae5','mfe5']].to_csv(OUT/'hazard_overlay_events.csv',index=False)
    return out

def main():
    q=pd.read_csv(BASE/'equity_curve.csv'); c=pd.read_csv(BASE/'closed_positions.csv'); e=pd.read_csv(BASE/'entries.csv')
    q['ts']=pd.to_datetime(q.ts,utc=True); c['close_ts']=pd.to_datetime(c.close_ts,utc=True); c['open_ts']=pd.to_datetime(c.open_ts,utc=True)
    daily=period_table(q,c,'D','daily'); weekly=period_table(q,c,'W','weekly'); monthly=period_table(q,c,'M','monthly_v46')
    mtm=q.mtm.to_numpy(float); p=c.pnl_net.to_numpy(float)
    base={'start_equity':START,'end_mtm':float(q.iloc[-1].mtm),'return_pct':float((q.iloc[-1].mtm/START-1)*100),'entries':len(e),'closed':len(c),
          'win_rate':float((c.pnl_net>0).mean()),'pf':pf(c.pnl_net),'maxdd_pct':maxdd(q.mtm),'net_pnl':float(c.pnl_net.sum())}
    roll={}
    for n in [1,7,30]:
        r=rolling(q,n); roll[f'{n}d']={'n':len(r),'positive_rate':float((r>0).mean()),'median_pct':float(r.median()),'p05_pct':float(r.quantile(.05)),'worst_pct':float(r.min()),'best_pct':float(r.max())}
    out={'baseline':base,'calendar':{'daily':stat(daily),'weekly':stat(weekly),'monthly':stat(monthly)},'rolling':roll,'hazard_overlay_diagnostic':hazard_overlay(),
         'notes':['Baseline is exact C600 path-dependent replay output already generated by V30.','V44 hazard overlay is diagnostic on the frozen 1103 baseline entry set; it is NOT a compounded equity simulation because skipped trades would alter later opportunity paths.','Order-book V44 remains telemetry-only and is not applied to PnL.']}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
    print('SAVED',OUT)
if __name__=='__main__': main()
