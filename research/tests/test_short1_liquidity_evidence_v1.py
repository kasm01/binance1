"""Test SHORT bid-depth evidence as-only, no orders or external data."""
from orchestration.context.short1_liquidity_evidence import short_sell_book_evidence as verify
t=1794655800000
base=dict(available=True,source_event_ms=t+19800,
          observed_at_ms=t+19900,best_bid=8.0,best_ask=8.001,
          bid_qty_l1=10.0,bid_qty_5=100.0,spread_bps=1.25)
a=verify(base,decision_ms=t+20100,notional_usdt=30)
assert a['available'] and a['short_bid_l1_cover']
assert not a['can_authorize'] and a['liq_score'] is None
assert a['realized_slippage_bps'] is None
b=verify(base,decision_ms=t+20100,notional_usdt=100)
assert b['available'] and not b['short_bid_l1_cover'] and b['short_bid_l5_cover']
assert not verify(base,decision_ms=t+10000,notional_usdt=30)['available']
assert not verify(base,decision_ms=t+25000,notional_usdt=30)['available']
assert not verify(dict(base,bid_qty_5=-1),decision_ms=t+20100,notional_usdt=30)['available']
assert not verify(dict(base,best_bid=9),decision_ms=t+20100,notional_usdt=30)['available']
assert not verify(dict(base,source_event_ms=None),decision_ms=t+20100,notional_usdt=30)['available']
assert not verify(dict(base,available=False),decision_ms=t+20100,notional_usdt=30)['available']
assert not verify(dict(base,bid_qty_l1=float('nan')),decision_ms=t+20100,notional_usdt=30)['available']
print('PASS 10 bid-side amount/age/sanity tests; no Master approval invented')
