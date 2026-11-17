"""Stage28: bridge immutable Stage26 reservation -> actual SHORT/LONG exact broker binder.

RESEARCH ONLY. No order POST, no redis writes, no autonomous COMMIT or RELEASE.
Production broker auth and atomic escrow final safety are not implemented.
"""
from __future__ import annotations
from decimal import Decimal,InvalidOperation
from typing import Any,Callable,Mapping
from orchestration.risk.bound_exchange_reconcile_v1 import bind

_ALLOWED={"LONG1":("NEARUSDT","BUY"),"SHORT1":("INJUSDT","SELL")}

def _positive(value:Any)->bool:
    try:
        if value is None or isinstance(value,bool):return False
        q=Decimal(str(value))
        return q.is_finite() and q>0
    except (InvalidOperation,TypeError,ValueError):return False

def assess_stage28(*,reservation:Mapping[str,Any]|None,
  ledger:Mapping[str,Any]|None,
  read_order:Callable,read_fills:Callable,read_positions:Callable,
  read_open_orders:Callable,read_open_algo:Callable,
  expected_epoch:int,expected_fence:int,now_ms:int,
  max_capture_sec:float=12.0)->dict:
    base={"state":"UNKNOWN_FREEZE","reason":"unverified",
          "can_submit":False,"can_resubmit":False,"can_commit":False,
          "can_release":False,"live_order_authorized":False,
          "external_broker_proof_authenticated":False}
    def deny(why:str,**extra)->dict:
        return dict(base,reason=why,**extra)
    if not isinstance(reservation,Mapping) or not isinstance(ledger,Mapping):
        return deny("reservation_or_ledger_missing")
    strategy=reservation.get("strategy")
    if strategy not in _ALLOWED:return deny("strategy_invalid")
    sym,side=_ALLOWED[strategy]
    reserved_direction="long" if side=="BUY" else "short"
    if reservation.get("symbol")!=sym or reservation.get("side")!=reserved_direction:
        return deny("reservation_strategy_side_mismatch")
    if ledger.get("owner")!=strategy or ledger.get("symbol")!=sym or ledger.get("side")!=side:
        return deny("ledger_strategy_side_mismatch")
    if type(expected_epoch) is not int or type(expected_fence) is not int or (
        expected_epoch<=0 or expected_fence<=0):
        return deny("epoch_or_fence_invalid")
    if type(reservation.get("epoch")) is not int or type(reservation.get("fence")) is not int:
        return deny("reservation_epoch_or_fence_missing")
    if reservation["epoch"]!=expected_epoch or reservation["fence"]!=expected_fence:
        return deny("stale_or_foreign_fence")
    for name in ("epoch","fence"):
        if type(ledger.get(name)) is not int or ledger[name]!=reservation[name]:
            return deny("ledger_epoch_or_fence_mismatch")
    if type(now_ms) is not int or now_ms<=0:
        return deny("invalid_current_clock")
    expiry=reservation.get("expires_at")
    if type(expiry) is not int or now_ms>expiry:
        # Expired lease does NOT make an UNKNOWN submitted order safe to retry.
        return deny("reservation_expired_reconciliation_required")
    if reservation.get("request")!=ledger.get("intent_id") or not reservation.get("request"):
        return deny("intent_request_mismatch")
    if reservation.get("owner")!=ledger.get("reservation_owner") or not reservation.get("owner"):
        return deny("reservation_owner_mismatch")
    if reservation.get("clientid")!=ledger.get("client_order_id") or not reservation.get("clientid"):
        return deny("client_order_id_mismatch")
    if not isinstance(reservation.get("digest"),str) or len(reservation["digest"])!=64:
        return deny("unbound_broker_digest")
    if not _positive(reservation.get("approved")):
        return deny("reserved_notional_invalid")
    # Bind the same immutable reservation ID to the journal and source evidence.
    reservation_id=f"epoch-{expected_epoch}-fence-{expected_fence}"
    if ledger.get("reservation_id")!=reservation_id:
        return deny("ledger_reservation_id_mismatch")
    projected={"request_id":reservation["request"],
               "reservation_id":reservation_id,"strategy_id":strategy,
               "symbol":sym}
    try:
        evidence=bind(reservation=projected,ledger=dict(ledger),
            read_order=read_order,read_fills=read_fills,
            read_positions=read_positions,read_open_orders=read_open_orders,
            read_open_algo=read_open_algo,max_capture_sec=max_capture_sec)
    except Exception:
        return deny("broker_reconciler_failed")
    if not isinstance(evidence,dict):
        return deny("broker_evidence_bad_type")
    evidence_state=str(evidence.get("state") or "UNKNOWN_FREEZE")
    # This is *evidence only*. Never automatically mutate the central lease
    # or interpret recovered fills as permission to resubmit.
    return dict(base,state=evidence_state,
      reason=str(evidence.get("reason") or "unverified"),
      broker_evidence_observed=evidence_state in (
          "PROTECTED_EVIDENCE_ONLY","FLAT_EVIDENCE_ONLY",
          "RECOVERED_FILL_PENDING_JOURNAL"),
      needs_human_or_authoritative_reconciler=True)
