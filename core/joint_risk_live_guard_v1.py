"""Shared LONG1/SHORT1 LIVE OPEN safety boundary — deny until live authority exists.

Both strategy executors import identical implementation. Do not confuse the
current DRY central-risk responder's READY/BYPASS with LIVE authority. The only
source of live authorization will be a separately reviewed independent,
authenticated, atomic risk reservation + final-safety verifier. NOT PRESENT.
No env switch or metadata can grant live open access here.
Existing position management, stop attachment, reduce-only closes are outside
this NEW-OPEN guard.
"""
from __future__ import annotations
from dataclasses import dataclass

ASSIGNMENTS={"LONG1":("NEARUSDT","long"),"SHORT1":("INJUSDT","short")}

@dataclass(frozen=True)
class JointOpenDecision:
    allow:bool
    reason:str
    live_order_authorized:bool=False

def check_new_open(*,strategy_id:str,symbol:str,side:str,dry_run:bool)->JointOpenDecision:
    strategy=str(strategy_id or "").strip().upper()
    pair=(str(symbol or "").strip().upper(),str(side or "").strip().lower())
    if strategy not in ASSIGNMENTS or pair!=ASSIGNMENTS[strategy]:
        return JointOpenDecision(False,"joint_risk_strategy_assignment_invalid")
    if bool(dry_run):
        # Only a DRY_RUN engine with independent guarantees that no order is
        # submitted may proceed; this cannot be used as a live permit.
        return JointOpenDecision(True,"joint_risk_offline_shadow_only")
    return JointOpenDecision(False,"joint_risk_live_authority_not_connected")
