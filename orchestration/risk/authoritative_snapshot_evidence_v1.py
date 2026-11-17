"""Read-only independent Binance/central-risk snapshot consistency checker.

This is DATA EVIDENCE only, never a risk authorization. Actual reserve headroom
must use trusted MTM snapshot + DD mode, full pending/committed positions,
available margin, fees/slippage and live orders and STOP/ALGO reconciliation.
"""
from __future__ import annotations
import math,time
from decimal import Decimal,InvalidOperation

def dec(v):
    try:
        z=Decimal(str(v))
        return z if z.is_finite() else None
    except (TypeError,ValueError,InvalidOperation):
        return None

def verify(account,positions,local_snapshot,*,now_sec=None,max_snapshot_age_sec=120,
           max_capture_age_sec=15,elapsed_sec=0,orders_known=False):
    res=dict(schema='risk_snapshot_crosscheck.v1',state='UNKNOWN',
       can_authorize_new_reservation=False,can_auto_commit=False,can_auto_release=False,
       reason='unverified',risk_mode=None,equity_usdt=None,available_balance_usdt=None,
       account_position_count=None,local_position_count=None)
    if not isinstance(account,dict) or not isinstance(positions,list) or not isinstance(local_snapshot,dict):
        return dict(res,reason='missing_source')
    if elapsed_sec is None or not math.isfinite(float(elapsed_sec)) or elapsed_sec<0 or elapsed_sec>max_capture_age_sec:
        return dict(res,reason='slow_or_invalid_capture')
    t=now_sec if now_sec is not None else time.time()
    snapts=dec(local_snapshot.get('ts'))
    if snapts is None or snapts>dec(t)+Decimal('2') or dec(t)-snapts>dec(max_snapshot_age_sec):
        return dict(res,reason='risk_snapshot_stale_or_future')
    acct=(local_snapshot.get('account') or {})
    if not isinstance(acct,dict):
        return dict(res,reason='malformed_central_account')
    names=('totalWalletBalance','totalUnrealizedProfit','totalMarginBalance',
           'availableBalance','totalInitialMargin','totalOpenOrderInitialMargin')
    vals={k:dec(account.get(k)) for k in names}
    if any(v is None for v in vals.values()):
        return dict(res,reason='broker_account_metrics_missing')
    wallet=vals['totalWalletBalance'];pnl=vals['totalUnrealizedProfit']
    balance=vals['totalMarginBalance'];available=vals['availableBalance']
    initial=vals['totalInitialMargin'];orders_margin=vals['totalOpenOrderInitialMargin']
    if wallet<0 or balance<=0 or available<0 or initial<0 or orders_margin<0:
        return dict(res,reason='invalid_or_unfunded_margin')
    if abs(wallet+pnl-balance)>Decimal('.02'):
        return dict(res,reason='account_margin_equation_mismatch')
    local_equity=dec(acct.get('mtm_equity'))
    local_wallet=dec(acct.get('wallet_balance'))
    local_avail=dec(acct.get('available_balance'))
    if None in (local_equity,local_wallet,local_avail):
        return dict(res,reason='local_equity_missing')
    if any(abs(a-b)>Decimal('0.02') for a,b in
           ((local_equity,balance),(local_wallet,wallet),(local_avail,available))):
        return dict(res,reason='central_vs_broker_equity_mismatch')
    seen=set();open_legs=[]
    for x in positions:
        if not isinstance(x,dict) or not x.get('symbol') or 'positionAmt' not in x:
            return dict(res,reason='position_row_malformed')
        amount=dec(x['positionAmt'])
        if amount is None:return dict(res,reason='position_nonfinite')
        pair=(x['symbol'],x.get('positionSide','BOTH'))
        if pair in seen:return dict(res,reason='duplicate_position_leg')
        seen.add(pair)
        if amount!=0:
            open_legs.append(dict(symbol=str(x['symbol']),side=str(x.get('positionSide','BOTH')),
                                  qty_abs=str(abs(amount))))
    local_positions=local_snapshot.get('positions')
    if not isinstance(local_positions,list):
        return dict(res,reason='local_positions_malformed')
    local_nonzero=[x for x in local_positions if isinstance(x,dict) and dec(x.get('notional')) is not None and dec(x.get('notional'))>0]
    if len(local_nonzero)!=len(local_positions):
        return dict(res,reason='local_position_entry_invalid')
    if len(open_legs)!=len(local_positions):
        return dict(res,reason='position_count_mismatch')
    if open_legs:
        # Matching quantities/side/mark notional is needed to enter VERIFIED,
        # so nonzero positions stay UNKNOWN in this V1 checker.
        return dict(res,reason='open_positions_need_strict_qty_notional_match',
                    account_position_count=len(open_legs),local_position_count=len(local_positions))
    profile=(local_snapshot.get('portfolio_state') or {})
    health=(local_snapshot.get('exposure_health') or {})
    if not isinstance(profile,dict) or not isinstance(health,dict):
        return dict(res,reason='risk_profile_unavailable')
    mode=str(profile.get('mode') or '')
    if mode not in ('NORMAL','REDUCED','RECOVERY'):
        return dict(res,reason='invalid_dd_mode')
    if health.get('status')!='OK':
        return dict(res,reason='exposure_health_not_ok')
    peak=dec(profile.get('equity_peak'));dd=dec(profile.get('drawdown_ratio'))
    if peak is None or dd is None or peak<balance or peak<=0:
        return dict(res,reason='invalid_peak_dd')
    if abs((balance/peak)-1-dd)>Decimal('.003'):
        return dict(res,reason='peak_dd_inconsistent')
    # Other assets, pending orders, unconfirmed intent journal and stop ownership
    # have not been fully proven; keep can_authorize_new_reservation=False.
    return dict(res,state='CONSISTENT_READONLY',
       reason=('consistent_equity_positions_but_order_authority_incomplete' if not orders_known
               else 'consistent_equity_positions_but_ledger_ownership_incomplete'),
       risk_mode=mode,equity_usdt=str(balance),
       available_balance_usdt=str(available),wallet_balance_usdt=str(wallet),
       open_order_initial_margin_usdt=str(orders_margin),
       account_position_count=len(open_legs),local_position_count=len(local_positions),
       central_snapshot_age_seconds=round(float(dec(t)-snapts),2))
