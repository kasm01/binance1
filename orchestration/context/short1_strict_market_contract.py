"""Fail-closed pre-Master SHORT1-INJ provenance and market-data completeness.

This is a required input-contract validator, NOT an authentication/signature
system and NOT model promotion. Do not derive missing whale/liquidity readings
from neutral default values or allow old target scores into the new lifecycle.
"""
from __future__ import annotations
from datetime import datetime,timezone
import math,time,os

FIELDS=('price','spread_pct','atr_pct','liq_score','whale_dir','whale_score',
        'money_flow_15m','money_flow_60m','price_return_15m')
NUMERIC=tuple(k for k in FIELDS if k!='whale_dir')
PROVIDER='short_tp125_sl299_v7'
TARGET='short_tp125_sl299_netpositive_60m_v7'
def scoped(item):
    return str(item.get('symbol') or '').upper()=='INJUSDT' and str(item.get('side') or '').lower() in ('short','sell')
def check(item, *, now_ms=None, max_age_ms=8*60000):
    """A deliberately conservative SHORT1-specific gate before score gating."""
    if not isinstance(item,dict):return False,'invalid_candidate'
    if not scoped(item):return True,'outside_short1_scope'
    try:
        sid=str(item.get('source_signal_event_id') or '')
        ms=int(item.get('source_completed_close_ms') or 0)
        event_ts=str(item.get('ts_utc') or '')
        dt=datetime.fromisoformat(event_ts.replace('Z','+00:00'))
        if dt.tzinfo is None:raise ValueError('no timezone')
        event_ms=int(dt.timestamp()*1000)
        now=int(now_ms if now_ms is not None else time.time()*1000)
        if not sid or not ms or ms%180000!=179999:
            return False,'source_id_or_completed_candle_invalid'
        if ms>event_ms or event_ms>now+2000 or (now-ms)>max_age_ms:
            return False,'source_not_fresh'
        # Also bind the source event publication time to its claimed candle.
        if event_ms-ms>max_age_ms:
            return False,'event_too_late'
    except (ValueError,TypeError,OverflowError):
        return False,'source_timestamp_invalid'
    if (item.get('model_provider')!=PROVIDER or
        item.get('model_target')!=TARGET):
        return False,'model_target_not_approved'
    # Do NOT trust a promotion flag supplied in a Redis event. Require an
    # independent local operator policy. OFF by default, and NOT in .env.
    if os.environ.get('SHORT1_V7_PRODUCTION_APPROVED','0') != '1':
        return False,'operator_model_promotion_disabled'
    if item.get('model_promotion_authorized') is not True:
        return False,'model_not_promoted'
    try:
        s=float(item['model_score'])
        threshold=float(item['model_probability_threshold'])
        if not math.isfinite(s) or not math.isfinite(threshold):
            return False,'nonfinite_model_score'
        if not (0<=s<=1 and .01<threshold<=1 and s>=threshold):
            return False,'source_score_below_threshold'
    except (ValueError,TypeError,KeyError):
        return False,'model_score_contract_missing'
    # Normalizer may have already defaulted the fields to zero. Require
    # evidence keys captured directly BEFORE normalization, not just values.
    present=item.get('source_field_presence')
    if not isinstance(present,list):
        return False,'source_field_presence_missing'
    missing=[k for k in FIELDS if k not in present or k not in item]
    if missing:return False,'source_market_fields_missing:'+','.join(missing)
    try:
        for k in NUMERIC:
            v=float(item[k])
            if not math.isfinite(v):return False,'nonfinite_market_field:'+k
        if float(item['price'])<=0 or float(item['atr_pct'])<0 or float(item['spread_pct'])<0:
            return False,'invalid_price_or_volatility'
        if not 0<=float(item['liq_score'])<=1 or not 0<=float(item['whale_score'])<=1:
            return False,'invalid_liquidity_or_whale_score'
        if any(abs(float(item[x]))>1 for x in ('money_flow_15m','money_flow_60m')):
            return False,'invalid_flow'
        if str(item['whale_dir']).lower() not in ('short','long','none'):
            return False,'invalid_whale_direction'
        if str(item['whale_dir']).lower()=='none' and item.get('whale_observation_status')!='confirmed_neutral':
            return False,'unverified_neutral_whale'
    except (ValueError,TypeError,KeyError):
        return False,'invalid_market_fields'
    # Still only source completeness, never a cryptographic permission.
    return True,'short1_source_contract_complete'
