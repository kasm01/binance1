"""Durable close evidence candidate queue, isolated research only.

A candidate is not an authoritative recovery win. Queue stores summary,
immutable reservation generation and a SHA-256 digest, but never issues a
CLOSE_EVENT to risk mode journal. No risk promotion under any circumstances.
"""
from __future__ import annotations
import hashlib,json
LUA=r'''
-- KEYS: mode_state, pending, events, seen, risk_gate
-- ARGV: eventid, intentid, generation, modeepoch, label, netpnl, digest
local id,iid,gen=ARGV[1],ARGV[2],ARGV[3]
local expected=tonumber(ARGV[4])
local label,pnl,digest=ARGV[5],ARGV[6],ARGV[7]
local function ans(state,reason)
 return cjson.encode({state=state,reason=reason,
 recovery_success_credit=false,mode_change_authorized=false,
 live_order_authorized=false})
end
if id=='' or iid=='' or gen=='' or not expected or expected<1 or
 #digest~=64 or label=='' then return ans('REJECT','invalid_candidate') end
for _,pair in ipairs({{KEYS[1],'string'},{KEYS[2],'hash'},
 {KEYS[3],'stream'},{KEYS[4],'hash'},{KEYS[5],'string'}}) do
 local typ=redis.call('TYPE',pair[1])['ok']
 if typ~='none' and typ~=pair[2] then return ans('REJECT','wrong_storage_type') end
end
local raw=redis.call('GET',KEYS[1])
if not raw then return ans('REJECT','mode_state_unavailable') end
local mode=cjson.decode(raw)
if mode.epoch~=expected then return ans('REJECT','mode_epoch_changed') end
if mode.source_gen~=gen then return ans('REJECT','source_generation_mismatch') end
local payload=iid..'|'..gen..'|'..tostring(expected)..'|'..label..'|'..digest
local seen=redis.call('HGET',KEYS[4],id)
if seen then
 if seen==payload then return ans('READY','idempotent_candidate') end
 return ans('REJECT','candidate_event_conflict')
end
local record={event_id=id,intent_id=iid,generation=gen,
 epoch=expected,label=label,net_pnl=pnl,
 digest=digest,authority='NONE'}
-- Append as a pending review, regardless of apparent positive pnl.
redis.call('SET',KEYS[5],'CLOSED')
redis.call('HSET',KEYS[2],id,cjson.encode(record))
redis.call('HSET',KEYS[4],id,payload)
redis.call('XADD',KEYS[3],'*','event_id',id,'intent_id',iid,
 'mode_epoch',tostring(expected),'label',label,'digest',digest)
return ans('READY','candidate_queued_no_risk_promotion')
'''
class IsolatedCloseCandidateQueue:
 def __init__(self,r,mode_state_key,prefix='research:close_review:v1'):
  self.r=r
  self.keys=[mode_state_key,prefix+':pending',prefix+':events',
    prefix+':seen',prefix+':gate']
 def enqueue(self,event_id,intent_id,generation,mode_epoch,evidence):
  if not isinstance(evidence,dict):raise ValueError('evidence_required')
  label=evidence.get('state','UNKNOWN_FREEZE')
  digest=hashlib.sha256(json.dumps(evidence,sort_keys=True,
        separators=(',',':')).encode()).hexdigest()
  return json.loads(self.r.eval(LUA,len(self.keys),*self.keys,
    event_id,intent_id,generation,mode_epoch,label,
    evidence.get('net_pnl_usdt') or '',digest))
