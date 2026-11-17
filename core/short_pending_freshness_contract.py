"""Pure SHORT PENDING model provenance validation.

Used to link new authenticated upstream intents to an existing PENDING.
Old stored scores alone cannot open a new order.
"""
from __future__ import annotations
import math
from typing import Mapping,Any

def check_independent_model_revalidation(origin:Mapping[str,Any], candidate:Mapping[str,Any], *,now_ms:int):
    try:
        oid=str(origin['intent_id'])
        nid=str(candidate['intent_id'])
        provider=str(origin['model_provider'])
        source_event_id=str(origin['source_signal_event_id'])
        new_event_id=str(candidate['source_signal_event_id'])
        source_ms=int(origin['completed_close_ms'])
        new_ms=int(candidate['completed_close_ms'])
        score=float(candidate['model_score'])
        threshold=float(candidate['model_score_gate_threshold'])
        confirmed=bool(candidate['model_score_gate_passed'])
    except (KeyError,TypeError,ValueError):
        return False,'missing_provenance_contract'
    if (not oid or not nid or oid==nid or not source_event_id or not new_event_id
            or source_event_id==new_event_id or not provider
            or provider!=str(candidate.get('model_provider') or '')):
        return False,'not_independent_or_provider_mismatch'
    if not (0<source_ms<new_ms<now_ms) or now_ms-new_ms>5*60*1000:
        return False,'model_bar_not_new_completed_and_fresh'
    if not math.isfinite(score) or not math.isfinite(threshold) or not (0<=score<=1 and 0<threshold<=1):
        return False,'invalid_model_score_contract'
    if not confirmed or score<threshold:return False,'model_score_gate_failed'
    return True,'independent_new_model_ready'
