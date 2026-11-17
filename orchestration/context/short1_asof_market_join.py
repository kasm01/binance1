"""Read-only, strict as-of Market Context joining for SHORT1 INJ shadow research.

No Master READY flag, no synthetic whale, no model promotion. Prefer source
timestamps over Redis stream append times. If a source is stale or later than
decision time, omit it instead of considering it neutral.
"""
from __future__ import annotations
import bisect,math

def finite_positive(value):
    try:
        v=float(value)
        return v if math.isfinite(v) and v>0 else None
    except (TypeError,ValueError):
        return None

def finite(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None

def build_indexes(v44_events):
    """Published V44 20s measurements. Timestamp is MARKET observation time."""
    index=[]
    for evt in v44_events:
        if evt.get('symbol')!='INJUSDT':
            continue
        ob=evt.get('orderbook_20s')
        if not isinstance(ob,dict) or ob.get('available') is not True:
            continue
        try:
            observed=int(ob.get('observed_at_ms'))
            source=int(ob.get('source_event_ms'))
            published=int(float(evt['ts'])*1000)
        except (TypeError,ValueError,KeyError):
            continue
        if not (0<source<=observed<=published+2000):
            continue
        # prevent reading event that was published AFTER the model signal,
        # even if the underlying market data was timestamped before it.
        index.append((published,observed,source,ob,evt))
    index.sort(key=lambda z:z[0])
    return index,[z[0] for z in index]

def join(v6,index,publish_times,*,max_orderbook_age_ms=120000,max_oi_age_ms=900000):
    if v6.get('symbol')!='INJUSDT':
        return {'state':'REJECT_SOURCE','reason':'wrong_symbol'}
    try:
        now_ms=int(float(v6['ts'])*1000)
        bar_ms=int(v6['completed_close_time'])
    except (TypeError,ValueError,KeyError):
        return {'state':'REJECT_SOURCE','reason':'missing_event_time'}
    if bar_ms>now_ms or now_ms-bar_ms>180000:
        return {'state':'REJECT_SOURCE','reason':'stale_model_event'}
    ctx=v6.get('entry_market_context')
    ft=v6.get('features')
    conf=v6.get('confirmation')
    result=dict(state='INCOMPLETE',symbol='INJUSDT',model_event_ms=now_ms,
        model_source_event_id=v6.get('event_id'),
        model_completed_close_ms=bar_ms,
        available_fields=[],missing_fields=[],field_evidence={},source='READ_ONLY_ASOF_SHADOW')
    values={}
    def attach(name,value,source,source_ms,**extra):
        if value is None:
            return
        values[name]=value
        result['field_evidence'][name]=dict(source=source,source_ms=int(source_ms),**extra)
    if isinstance(conf,dict):
        close=finite_positive(conf.get('curr_close'))
        attach('price',close,'v6.partial_1m_snapshot',now_ms)
    if isinstance(ft,dict):
        atr=finite(ft.get('atr14_pct'))
        attach('atr_pct',atr if atr is not None and 0<=atr<1 else None,'v6.completed_3m_features',bar_ms)
    if isinstance(ctx,dict) and ctx.get('available') is True:
        pr=ctx.get('price_response') or {}
        flow=ctx.get('money_flow') or {}
        oi=ctx.get('open_interest') or {}
        attach('price_return_15m',finite(pr.get('return_15m')),'v6.completed_1m_context',now_ms)
        attach('money_flow_15m',finite(flow.get('flow_15m')),'v6.completed_1m_context',now_ms)
        if oi.get('available') is True:
            try:
                oits=int(oi['source_ts_ms'])
                if 0<=now_ms-oits<=max_oi_age_ms:
                    attach('oi_delta_15m',finite(oi.get('oi_delta_15m')),'v6.prior_oi_cache',oits)
            except (ValueError,TypeError,KeyError):
                pass
    attach('money_flow_60m',finite(v6.get('money_flow_60m')),'v6.completed_1m_flow',now_ms)
    pos=bisect.bisect_right(publish_times,now_ms)-1
    if pos>=0:
        pub_ms,observed_ms,source_ms,ob,evt=index[pos]
        age=now_ms-source_ms
        if 0<=age<=max_orderbook_age_ms and pub_ms<=now_ms and observed_ms<=now_ms:
            spread_bps=finite(ob.get('spread_bps'))
            bid=finite(ob.get('bid_qty_5'))
            ask=finite(ob.get('ask_qty_5'))
            if spread_bps is not None and spread_bps>=0:
                attach('spread_pct',spread_bps/10000,'v44.observed_orderbook_20s',
                       source_ms,published_ms=pub_ms,age_ms=age)
            if bid is not None and ask is not None and bid>=0 and ask>=0 and bid+ask>0:
                # Raw best-5 depth is evidence; NOT canonical normalized liq_score.
                attach('bid_qty_5',bid,'v44.observed_orderbook_20s',source_ms,
                       published_ms=pub_ms,age_ms=age)
                attach('ask_qty_5',ask,'v44.observed_orderbook_20s',source_ms,
                       published_ms=pub_ms,age_ms=age)
    # Do not infer whale score from orderbook imbalance or call it 'neutral'.
    # No canonical liq_score recipe has been approved and tested.
    expected=('price','atr_pct','spread_pct','liq_score','whale_dir','whale_score',
              'money_flow_15m','money_flow_60m','price_return_15m')
    result.update(values=values,
        available_fields=sorted(k for k in expected if k in values),
        missing_fields=sorted(k for k in expected if k not in values),
        state='BLOCK_MASTER' if any(k not in values for k in expected) else 'COMPLETE_SOURCE_DATA_ONLY')
    return result

def build_pending_20s_index(v44_events):
    """Index actual 20-second public book observations by minute and source.
    V44 source candidate may be V5, not V6; minute and symbol join only
    produces observation evidence, never new-model proof.
    """
    groups={}
    for e in v44_events:
        if e.get('symbol')!='INJUSDT' or e.get('stage')!='20S_HAZARD':
            continue
        ob=e.get('orderbook_20s')
        if not isinstance(ob,dict) or ob.get('available') is not True:
            continue
        try:
            minute=int(e['minute_ms'])
            published=int(float(e['ts'])*1000)
            observed=int(ob['observed_at_ms'])
            source=int(ob['source_event_ms'])
        except (KeyError,TypeError,ValueError):
            continue
        if not (minute+19000<=source<=observed<=published+2000<=minute+28000):
            continue
        if e.get('routing')!='OFF' or e.get('policy_effect') not in ('TELEMETRY_ONLY','NONE',None):
            continue
        groups.setdefault(minute,[]).append((published,source,ob,e))
    return groups

def evaluate_pending_20s(v6,groups,*,max_orderbook_age_ms=2000):
    """Research evidence: would V44 be FRESH at 20s, AFTER V6 model event?

    DOES NOT imply SHORT1 Entry Confirmation is still READY at 20s.
    """
    try:
        ms=int(float(v6['ts'])*1000)
        minute=int((v6.get('confirmation') or {})['current_open_time'])
        if v6.get('symbol')!='INJUSDT' or minute>ms:
            return dict(state='REJECT_SOURCE',reason='bad_event_or_minute')
    except (ValueError,TypeError,KeyError):
        return dict(state='REJECT_SOURCE',reason='missing_partial_1m_open')
    choices=groups.get(minute,[])
    matching=[row for row in choices if row[0]>ms and row[0]-ms<=25000
              and row[0]-row[1]<=max_orderbook_age_ms]
    if not matching:
        return dict(state='PENDING_NO_FRESH_V44_20S',
                    model_ms=ms,minute_ms=minute)
    pub,src,ob,e=min(matching,key=lambda row:row[0])
    spread=finite(ob.get('spread_bps'))
    bid=finite(ob.get('bid_qty_5'))
    ask=finite(ob.get('ask_qty_5'))
    if spread is None or spread<0 or bid is None or ask is None or bid<0 or ask<0:
        return dict(state='PENDING_BAD_OBSERVATION',model_ms=ms,minute_ms=minute)
    return dict(state='PENDING_BOOK_EVIDENCE_AVAILABLE',
        model_ms=ms,minute_ms=minute,book_source_ms=src,
        book_publish_ms=pub,delay_after_model_ms=pub-ms,
        book_age_on_publish_ms=pub-src,
        spread_pct=spread/10000,
        bid_qty5=bid,ask_qty5=ask,
        model_confirm_at_20s='NOT_CHECKED',
        independent_model_event_at_20s='NOT_CHECKED',
        whale_score='MISSING',liquidity_score='MISSING',
        executor_authorized=False)
