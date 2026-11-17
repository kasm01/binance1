"""SHORT1 shadow-only order-book bid capacity evidence, NOT canonical liq_score.

SHORT market sell consumes BIDS. Aggregated L5 has no individual price levels;
therefore this cannot estimate executable VWAP, fees, or actual slippage.
"""
from math import isfinite

def _number(v):
    try:
        z=float(v)
        return z if isfinite(z) else None
    except (ValueError,TypeError):
        return None

def short_sell_book_evidence(book, *, decision_ms, notional_usdt, max_age_ms=2000):
    base=dict(available=False,can_authorize=False,liq_score=None,
       realized_slippage_bps=None,entry_notional_usdt=notional_usdt)
    if not isinstance(book,dict) or book.get('available') is not True:
        return dict(base,reason='missing_order_book')
    try:
        src=int(book['source_event_ms'])
        observed=int(book['observed_at_ms'])
        now=int(decision_ms)
    except (ValueError,TypeError,KeyError):
        return dict(base,reason='invalid_source_timestamp')
    if src<=0 or not (src<=observed<=now) or now-src>max_age_ms:
        return dict(base,reason='stale_or_future_book')
    bid=_number(book.get('best_bid'));ask=_number(book.get('best_ask'))
    l1=_number(book.get('bid_qty_l1'));l5=_number(book.get('bid_qty_5'))
    spread=_number(book.get('spread_bps'));amount=_number(notional_usdt)
    if (None in (bid,ask,l1,l5,spread,amount) or
        bid<=0 or ask<bid or l1<0 or l5<l1 or spread<0 or amount<=0):
        return dict(base,reason='invalid_bid_capacity')
    l1_usdt=bid*l1;l5_usdt=bid*l5
    return dict(base,available=True,reason='valid_quote_depth_observation',
       source_ms=src,source_age_ms=now-src,
       short_bid_l1_cover=bool(l1_usdt>=amount),
       short_bid_l5_cover=bool(l5_usdt>=amount),
       short_bid_l1_notional_usdt=round(l1_usdt,6),
       short_bid_l5_notional_usdt=round(l5_usdt,6),
       spread_bps=spread,entry_notional_usdt=amount)
