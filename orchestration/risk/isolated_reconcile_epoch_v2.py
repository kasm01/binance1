"""Research-only monotonic reconciliation journal with sticky risk severity.

Never a production risk gate. Epochs reflect START order of observations, not
completion order; a delayed positive observation cannot supersede a newer
negative one. Every distinct result is appended; all gates remain CLOSED.
Mock/GET evidence is not independently authenticated. No COMMIT/RELEASE.
"""
from __future__ import annotations
import json,hashlib,time,math
from orchestration.risk.bound_exchange_reconcile_v1 import bind

# High severity is sticky; only an independent, authenticated and fully
# reconciled production recovery could clear it (not implemented here).
SEVERITY={
 'PROTECTED_EVIDENCE_ONLY':1,
 'FLAT_EVIDENCE_ONLY':1,
 'RECOVERED_FILL_PENDING_JOURNAL':3,
 'HOLD_RECONCILIATION':3,
 'UNKNOWN_FREEZE':4,
 'FREEZE_UNPROTECTED':5,
}
LUA=r'''
-- KEYS: gate, reservations, intents, epochcounter, current, event-id-index,
--       append-only events, hazard-outbox
-- ARGV: iid,gen,cid,event_id,epoch,label,severity,digest,started_ms,ended_ms
local iid,gen,cid,eventid=ARGV[1],ARGV[2],ARGV[3],ARGV[4]
local epoch=tonumber(ARGV[5])
local label=ARGV[6]
local severity=tonumber(ARGV[7])
local digest=ARGV[8]
local started=tonumber(ARGV[9])
local ended=tonumber(ARGV[10])
local function ans(st,why,cur)
 return cjson.encode({state=st,reason=why,current_epoch=cur or 0,
    order_authorized=false,commit_authorized=false,release_authorized=false})
end
if iid=='' or gen=='' or cid=='' or eventid=='' or not epoch or epoch<1
 or not severity or severity<1 or severity>5
 or not started or not ended or started<=0 or ended<started
 or ended-started>15000 or #digest~=64 then
 return ans('REJECT','invalid_audit_input')
end
for _,k in ipairs({KEYS[2],KEYS[3],KEYS[5],KEYS[6]}) do
 local t=redis.call('TYPE',k)['ok']
 if t~='none' and t~='hash' then return ans('REJECT','wrong_hash_type') end
end
for _,k in ipairs({KEYS[7],KEYS[8]}) do
 local t=redis.call('TYPE',k)['ok']
 if t~='none' and t~='stream' then return ans('REJECT','wrong_stream_type') end
end
local ct=redis.call('TYPE',KEYS[4])['ok']
if ct~='string' and ct~='none' then return ans('REJECT','wrong_counter_type') end
local maxepoch=tonumber(redis.call('GET',KEYS[4])) or 0
if epoch>maxepoch then return ans('REJECT','epoch_not_issued') end
local event_payload=iid..'|'..gen..'|'..cid..'|'..tostring(epoch)..'|'..label..'|'..digest
local existing=redis.call('HGET',KEYS[6],eventid)
if existing then
 if existing==event_payload then
  return ans('READY','idempotent_event_replay')
 end
 return ans('REJECT','event_id_conflict')
end
local record=redis.call('HGET',KEYS[3],iid)
if not record then return ans('REJECT','intent_not_found') end
local l=cjson.decode(record)
if l.intent_id~=iid or l.reservation_id~=gen or l.client_order_id~=cid
 then return ans('REJECT','intent_identity_mismatch') end
local reserve=redis.call('HGET',KEYS[2],l.owner)
if not reserve then return ans('REJECT','reservation_missing') end
local r=cjson.decode(reserve)
if r.request_id~=iid or r.reservation_id~=gen or
   r.symbol~=l.symbol or r.strategy_id~=l.owner then
 return ans('REJECT','reservation_owner_mismatch') end
local latest_raw=redis.call('HGET',KEYS[5],iid)
local latest=latest_raw and cjson.decode(latest_raw) or nil
local applied=not latest or epoch>tonumber(latest.epoch)
local stale=not applied
local prior_max=latest and tonumber(latest.max_severity) or 0
local max_severity=math.max(prior_max,severity)
local record_out={epoch=epoch,label=label,severity=severity,
 max_severity=max_severity,event_id=eventid,digest=digest,started_ms=started,
 ended_ms=ended,gen=gen,client_id=cid}
-- A newer observation can replace the newest event pointer, but prior
-- critical severity is sticky even if the newer event appears positive.
if applied then
 redis.call('HSET',KEYS[5],iid,cjson.encode(record_out))
elseif severity>prior_max then
 -- Late negative evidence cannot become the latest dated observation, but
 -- its higher hazard is still sticky at the current record.
 latest.max_severity=severity
 latest.late_hazard_seen=eventid
 redis.call('HSET',KEYS[5],iid,cjson.encode(latest))
end
redis.call('SET',KEYS[1],'CLOSED')
redis.call('HSET',KEYS[6],eventid,event_payload)
redis.call('XADD',KEYS[7],'*',
 'intent_id',iid,'event_id',eventid,'epoch',tostring(epoch),
 'state',label,'severity',tostring(severity),'digest',digest,
 'late',stale and '1' or '0')
-- Every serious observation, including late ones, is retained as a distinct
-- durable hazard. This is an outbox event, NOT guaranteed operator delivery.
if severity>=3 then
 redis.call('XADD',KEYS[8],'*',
  'intent_id',iid,'event_id',eventid,'epoch',tostring(epoch),
  'state',label,'severity',tostring(severity),
  'late',stale and '1' or '0')
end
return ans('READY',stale and 'stale_event_archived' or 'new_epoch_archived',
  applied and epoch or tonumber(latest.epoch))
'''
class IsolatedEpochAudit:
 def __init__(self,r,reserve_keys,prefix='research:reconcile:epochv2'):
  if len(reserve_keys)!=7:raise ValueError('expected_isolated_risk_keys')
  self.r=r
  self.risk=reserve_keys
  self.keys=[reserve_keys[0],reserve_keys[1],reserve_keys[4],
    prefix+':epoch_counter',prefix+':current',prefix+':event_ids',
    prefix+':events',prefix+':hazards']
 def begin(self):
  # Issue epoch BEFORE broker I/O, to prevent delayed positive rechecks
  # overwriting a newer negative recheck completed earlier.
  return self.r.incr(self.keys[3])
 def persist(self,*,intent_id,generation,client_order_id,event_id,epoch,
             result,started_ms,ended_ms):
  state=result.get('state','UNKNOWN_FREEZE')
  if state not in SEVERITY:state='UNKNOWN_FREEZE'
  severity=SEVERITY[state]
  digest=hashlib.sha256(json.dumps(result,sort_keys=True,
                     separators=(',',':')).encode()).hexdigest()
  return json.loads(self.r.eval(LUA,len(self.keys),*self.keys,
   intent_id,generation,client_order_id,event_id,epoch,state,severity,
   digest,started_ms,ended_ms))
 def inspect_and_persist(self,*,intent_id,event_id,broker,clock_ms=lambda:int(time.time()*1000)):
  try:
   epoch=self.begin()
   started=clock_ms()
   raw=self.r.hget(self.risk[4],intent_id)
   if not raw:
    return dict(state='UNKNOWN_FREEZE',reason='no_intent_to_bind',
      order_authorized=False,commit_authorized=False,release_authorized=False)
   ledger=json.loads(raw)
   reserve_raw=self.r.hget(self.risk[1],ledger.get('owner'))
   if not reserve_raw:
    return dict(state='UNKNOWN_FREEZE',reason='no_reservation_to_bind',
      order_authorized=False,commit_authorized=False,release_authorized=False)
   reserve=json.loads(reserve_raw)
   result=bind(reservation=reserve,ledger=ledger,
      read_order=broker.get_order,read_fills=broker.get_fills,
      read_positions=broker.get_positions,
      read_open_orders=broker.get_open_orders,
      read_open_algo=broker.get_open_algos)
   ended=clock_ms()
   if ended<started or ended-started>15000:
    result=dict(state='UNKNOWN_FREEZE',reason='broker_read_out_of_window')
   audit=self.persist(intent_id=intent_id,generation=ledger['reservation_id'],
     client_order_id=ledger['client_order_id'],event_id=event_id,epoch=epoch,
     result=result,started_ms=started,ended_ms=max(ended,started))
   return dict(audit=audit,evidence=result,order_authorized=False,
     commit_authorized=False,release_authorized=False)
  except Exception as exc:
   return dict(state='UNKNOWN_FREEZE',reason='reconciler_failed:'+type(exc).__name__,
    order_authorized=False,commit_authorized=False,release_authorized=False)
