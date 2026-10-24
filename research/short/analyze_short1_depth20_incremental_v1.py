#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import redis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

V44='telemetry:short1:inj:entry_hazard_orderbook_v44_outcomes'
OUTCOME='telemetry:short1:inj:model_shadow_v5_outcomes'
DEST=Path('research/short/results/SHORT1_DEPTH20_INCREMENTAL_V1.json')


def load(r):
    vv=[]
    for _,f in r.xrange(V44):
        try:
            o=json.loads(f.get('json','{}')); a=o.get('absorption_features_20s') or {}
            vv.append({'event_id':o.get('source_event_id'),'d20':float(a.get('depth20_support') or 0),'d5':float(a.get('depth5_support') or 0)})
        except Exception: pass
    oo=[]
    for _,f in r.xrange(OUTCOME):
        try:
            o=json.loads(f.get('json','{}'))
            oo.append({'event_id':o.get('event_id'),'ts':float(o.get('source_ts') or 0),'score':float(o.get('economic_probability') or 0),'net':float(o.get('net_return_pct') or 0),'confirmation':o.get('confirmation_state')})
        except Exception: pass
    d=pd.DataFrame(oo).merge(pd.DataFrame(vv),on='event_id')
    d['day']=pd.to_datetime(d.ts,unit='s',utc=True).dt.strftime('%Y-%m-%d'); d['win']=(d.net>0).astype(int)
    return d


def model():
    return Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(C=.2,max_iter=2000,class_weight='balanced',solver='liblinear',random_state=42))])


def eval_model(tr,te,cols,label):
    if len(tr)<100 or len(te)<20 or tr.win.nunique()<2 or te.win.nunique()<2:return None
    m=model().fit(tr[cols],tr.win); pt=m.predict_proba(tr[cols])[:,1]; p=m.predict_proba(te[cols])[:,1]
    auc=float(roc_auc_score(te.win,p)); th=float(np.quantile(pt,.8)); z=te[p>=th]
    pos=z.loc[z.net>0,'net'].sum(); neg=-z.loc[z.net<0,'net'].sum()
    return {'label':label,'auc':auc,'selected_n':int(len(z)),'selected_bps':float(z.net.mean()*1e4) if len(z) else None,'selected_pf':float(pos/neg) if neg>0 else (999.0 if pos>0 else None),'selected_win':float(z.win.mean()) if len(z) else None,'train_threshold':th}


def rank_rule(tr,te,top=True):
    q=float(tr.d20.quantile(.8 if top else .2)); z=te[te.d20>=q] if top else te[te.d20<=q]
    pos=z.loc[z.net>0,'net'].sum(); neg=-z.loc[z.net<0,'net'].sum()
    return {'direction':'top' if top else 'bottom','threshold':q,'selected_n':int(len(z)),'selected_bps':float(z.net.mean()*1e4) if len(z) else None,'selected_pf':float(pos/neg) if neg>0 else (999.0 if pos>0 else None),'selected_win':float(z.win.mean()) if len(z) else None}


def main(print_report: bool = True):
    d=load(redis.Redis(decode_responses=True)); days=sorted(d.day.unique()); out={'schema_version':'short1_depth20_incremental.v1','routing':'OFF','policy_effect':'TELEMETRY_ONLY','rows':int(len(d)),'days':days,'subsets':{}}
    for name,dd in [('ALL',d),('CONFIRM',d[d.confirmation=='CONFIRM'])]:
        tests=[]
        for i in range(2,len(days)):
            tr=dd[dd.day.isin(days[:i])]; te=dd[dd.day==days[i]]
            row={'day':days[i],'train_n':int(len(tr)),'test_n':int(len(te)),'models':[]}
            for label,cols in [('score',['score']),('d20',['d20']),('score+d20',['score','d20']),('score+d20+d5',['score','d20','d5'])]:
                z=eval_model(tr,te,cols,label)
                if z: row['models'].append(z)
            if len(tr)>=100 and len(te): row['d20_rules']=[rank_rule(tr,te,True),rank_rule(tr,te,False)]
            tests.append(row)
        out['subsets'][name]={'n':int(len(dd)),'tests':tests}

    # Promotion-style research gate. This report never routes orders.
    all_tests=[t for t in out['subsets']['ALL']['tests'] if any(m.get('label')=='d20' for m in t.get('models',[]))]
    conf_tests=[t for t in out['subsets']['CONFIRM']['tests'] if any(m.get('label')=='d20' for m in t.get('models',[]))]
    def _good(t):
        m=next((x for x in t.get('models',[]) if x.get('label')=='d20'),None)
        return bool(m and m.get('auc',0)>0.5 and (m.get('selected_bps') or -1)>0 and (m.get('selected_pf') or 0)>1.10)
    all_good=sum(_good(t) for t in all_tests); conf_good=sum(_good(t) for t in conf_tests)
    review_ok=(len(days)>=10 and len(conf_tests)>=5 and conf_good>=4 and len(all_tests)>=5 and all_good>=4)
    out['research_gate']={
        'status':'REVIEW_CANDIDATE' if review_ok else 'COLLECT_MORE',
        'minimum_calendar_days':10,'minimum_causal_test_days':5,'minimum_good_test_days':4,
        'all_causal_test_days':len(all_tests),'all_good_test_days':all_good,
        'confirm_causal_test_days':len(conf_tests),'confirm_good_test_days':conf_good,
        'routing':'OFF',
    }
    DEST.write_text(json.dumps(out,indent=2,allow_nan=False))
    if print_report: print(json.dumps(out,indent=2,allow_nan=False))
    return out

if __name__=='__main__': main()
