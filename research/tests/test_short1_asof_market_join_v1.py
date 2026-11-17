"""As-of/20s race tests. No Redis, no API calls, no trade execution."""
from orchestration.context.short1_asof_market_join import (
    build_indexes,join,build_pending_20s_index,evaluate_pending_20s)
BASE=1794654900000
def v6(ts=BASE+6280):
    return dict(symbol='INJUSDT',event_id='unique-source',
        ts=ts/1000,completed_close_time=BASE-1,
        confirmation={'current_open_time':BASE,'curr_close':7.92},
        features={'atr14_pct':.005},
        money_flow_60m=-.2,
        entry_market_context=dict(available=True,
            price_response={'return_15m':-.002},
            money_flow={'flow_15m':-.11},
            open_interest=dict(available=True,source_ts_ms=BASE-300000,oi_delta_15m=.002)))
def v44(*,publish=BASE+20050,source=BASE+19900,stage='20S_HAZARD',reported_minute=BASE):
    return dict(symbol='INJUSDT',stage=stage,
        minute_ms=reported_minute,ts=publish/1000,
        routing='OFF',policy_effect='TELEMETRY_ONLY',
        orderbook_20s=dict(available=True,
            observed_at_ms=publish-10,source_event_ms=source,
            spread_bps=1.2,bid_qty_5=123.0,ask_qty_5=234.0))
evt=v6()
assert join(evt,*build_indexes([v44()]))['values'].get('spread_pct') is None,'Future book cannot pass original 6sec model'
assert join(evt,*build_indexes([v44(publish=BASE-160000,source=BASE-160150)]))['values'].get('spread_pct') is None,'Stale book cannot pass'
x=join(evt,*build_indexes([]))
assert x['state']=='BLOCK_MASTER'
assert 'atr_pct' in x['available_fields']
assert {'whale_dir','whale_score','liq_score','spread_pct'}.issubset(x['missing_fields'])
pending=build_pending_20s_index([v44()])
a=evaluate_pending_20s(evt,pending)
assert a['state']=='PENDING_BOOK_EVIDENCE_AVAILABLE' and a['executor_authorized'] is False
assert a['delay_after_model_ms']==13770
assert evaluate_pending_20s(evt,build_pending_20s_index([v44(source=BASE+15000)]))['state']=='PENDING_NO_FRESH_V44_20S'
assert evaluate_pending_20s(evt,build_pending_20s_index([v44(publish=BASE+35000,source=BASE+34900)]))['state']=='PENDING_NO_FRESH_V44_20S'
assert evaluate_pending_20s(evt,build_pending_20s_index([v44(reported_minute=BASE-60000)]))['state']=='PENDING_NO_FRESH_V44_20S'
assert evaluate_pending_20s(evt,build_pending_20s_index([v44(stage='30S_REVALIDATION')]))['state']=='PENDING_NO_FRESH_V44_20S'
assert evaluate_pending_20s(evt,build_pending_20s_index([v44(source=BASE+20045)]))['state']=='PENDING_NO_FRESH_V44_20S' # source observed_at + <publication check? careful
assert not join(dict(v6(),symbol='SOLUSDT'),*build_indexes([]))['state']=='BLOCK_MASTER'
print('PASS 10 as-of market completeness and 20s no-future/no-stale/no-exec tests')
