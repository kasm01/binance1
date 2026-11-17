"""Fail-closed, read-only central risk sizing evidence, research only.

Matches currently deployed dry-run profile (BASE_REQUEST=.50,
operational=.92, REDUCED=.50, RECOVERY=.70, NORMAL=1.0).
Not an authorizer and NOT wired to V2 live/isolated reserve Lua.
Requires trusted per-call snapshot and independently reconciled exposures.
"""
from __future__ import annotations
from decimal import Decimal,InvalidOperation
import math,time

PROFILE='central_risk_enforce_dry.v1_h046_dd12_r06'
M={'NORMAL':Decimal('1.0'),'RECOVERY':Decimal('.70'),'REDUCED':Decimal('.50')}
def _d(x):
 try:
  v=Decimal(str(x))
  return v if v.is_finite() else None
 except (ValueError,TypeError,InvalidOperation):return None
def evaluate(evidence,*,strategy,now=None,max_age=120,market_multiplier=1):
 base=dict(state='UNKNOWN_FREEZE',reason='unverified',allowed_notional='0',
  order_authorized=False,reservation_authorized=False,profile_id=PROFILE)
 if not isinstance(evidence,dict):return dict(base,reason='no_evidence')
 if strategy not in ('LONG1','SHORT1'):return dict(base,reason='unknown_strategy')
 if evidence.get('state')!='CONSISTENT_READONLY':
  return dict(base,reason='broker_central_not_consistent')
 if evidence.get('can_authorize_new_reservation') is not False:
  return dict(base,reason='unsafe_authority_flag')
 if evidence.get('risk_mode') not in M:
  return dict(base,reason='invalid_risk_mode')
 balance=_d(evidence.get('equity_usdt'))
 available=_d(evidence.get('available_balance_usdt'))
 age=_d(evidence.get('central_snapshot_age_seconds'))
 mkt=_d(market_multiplier)
 if any(x is None for x in (balance,available,age,mkt)):
  return dict(base,reason='missing_or_nonfinite_balance_age_multiplier')
 if balance<=0 or available<0 or age<0 or age>_d(max_age) or mkt<=0 or mkt>1:
  return dict(base,reason='invalid_or_stale_evidence')
 # Position/regular order/algo/intent cross-check is NOT complete; despite
 # the budget being computed, order and reservation remain forbidden.
 if evidence.get('account_position_count')!=0 or evidence.get('local_position_count')!=0:
  return dict(base,reason='position_exposure_needs_reconciliation')
 if evidence.get('central_dry_pending_reservation_count')!=0:
  return dict(base,reason='pending_reservation_held')
 qty=balance*Decimal('.50')*Decimal('.92')*M[evidence['risk_mode']]*mkt
 # Hard caps: single side .5 equity, symbol .5, gross 1.0,
 # plus available-margin conversion is NOT justified; no real authority.
 if qty<10:
  return dict(base,state='REJECT_BUDGET',reason='below_min_order_notional',
     risk_mode=evidence['risk_mode'],
     illustrative_notional=str(qty))
 return dict(base,state='BUDGET_EVIDENCE_ONLY',
  reason='profile_sizing_only_not_executable',
  risk_mode=evidence['risk_mode'],
  illustrative_notional=str(qty),
  state_multiplier=str(M[evidence['risk_mode']]),
  base_ratio='0.50',operational_headroom='0.92')
