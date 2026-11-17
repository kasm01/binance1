"""Strict read-only Central Control hazard gate evidence, research-only.

Reads durable reservation, immutable intent, newest event pointer, hazard
stream and local receipts. NEVER opens trade gate or clears sticky hazard.
Race between reads means informational observation only. A real controller
needs an authenticated, atomic snapshot/ACL and independently verified mode.
"""
from __future__ import annotations
import json

def inspect(r,risk_keys,epoch_keys,consumer,*,intent_id):
 result=dict(state='FREEZE',reason='source_unverified',
             risk_severity=None,can_open=False,can_commit=False,
             can_release=False,can_restore_normal=False)
 try:
  gate=r.get(risk_keys[0])
  if gate!='CLOSED':return dict(result,reason='risk_gate_not_explicitly_closed')
  raw=r.hget(risk_keys[4],intent_id)
  if not raw:return dict(result,reason='intent_missing')
  intent=json.loads(raw)
  raw=r.hget(risk_keys[1],intent.get('owner',''))
  if not raw:return dict(result,reason='reservation_missing')
  reserve=json.loads(raw)
  if (reserve.get('request_id')!=intent_id or
      reserve.get('reservation_id')!=intent.get('reservation_id') or
      reserve.get('strategy_id')!=intent.get('owner')):
   return dict(result,reason='intent_reservation_mismatch')
  raw=r.hget(epoch_keys[4],intent_id)
  if not raw:return dict(result,reason='no_epoch_audit')
  audit=json.loads(raw)
  sev=audit.get('max_severity')
  if type(sev) is not int or sev<1 or sev>5:
   return dict(result,reason='hazard_severity_invalid')
  event_stream=epoch_keys[6];hazard_stream=epoch_keys[7]
  events=r.xrange(event_stream)
  hazards=r.xrange(hazard_stream)
  if not isinstance(events,list) or not isinstance(hazards,list):
   return dict(result,reason='event_sources_unavailable')
  current=[x for _,x in events if x.get('intent_id')==intent_id]
  relevant=[x for _,x in hazards if x.get('intent_id')==intent_id]
  if not current or str(audit['event_id']) not in {x.get('event_id') for x in current}:
   return dict(result,reason='epoch_current_not_in_event_log')
  if sev>=3 and not relevant:
   return dict(result,reason='critical_hazard_missing_from_outbox')
  if any(x.get('severity') not in ('3','4','5') for x in relevant):
   return dict(result,reason='invalid_hazard_outbox_severity')
  receipts=r.hgetall(consumer.receipts)
  delivered=set(receipts)
  missing=[x['event_id'] for x in relevant if x.get('event_id') not in delivered]
  # Safe response never clears the critical flag. Missing receipts make
  # urgent remediation still pending; local receipts are not external delivery.
  return dict(result,
     state='CRITICAL_FREEZE' if sev>=3 else 'REVIEW_FREEZE',
     reason='unacknowledged_risk_hazards' if missing else 'durable_hazards_review_required',
     risk_severity=sev,hazard_events=len(relevant),
     local_receipts=len(relevant)-len(missing),
     pending_local_receipts=len(missing),risk_mode_change_authorized=False)
 except Exception as exc:
  return dict(result,reason='read_failed:'+type(exc).__name__)
