"""Research-only durable hazard outbox consumption on isolated Redis.

XREADGROUP/XAUTOCLAIM for at-least-once processing. Atomic Lua writes one
durable LOCAL delivery receipt and XACKs the stream entry. This is NOT
exactly-once external delivery; no network/notification is sent. A supervisor
must independently verify stored receipt, source/epoch, pending state.
Never performs orders, risk release, gate reopen or production Redis writes.
"""
from __future__ import annotations
import json,time

ACK_LUA=r'''
-- KEYS hazard_stream,receipts,receipt_stream
-- ARGV group,streamid,event_id,intent_id,epoch,priority
local group,sid,eid,iid,epoch,priority=unpack(ARGV)
local function reply(st,msg)
 return cjson.encode({state=st,reason=msg,order_authorized=false,
 commit_authorized=false,release_authorized=false})
end
if group=='' or sid=='' or eid=='' or iid=='' or epoch=='' or priority=='' then
 return reply('REJECT','invalid_receipt_identity') end
for _,pair in ipairs({{KEYS[1],'stream'},{KEYS[2],'hash'},
 {KEYS[3],'stream'}}) do
 local typ=redis.call('TYPE',pair[1])['ok']
 if typ~='none' and typ~=pair[2] then return reply('REJECT','wrong_storage_type') end
end
local persisted=redis.call('HGET',KEYS[2],eid)
local payload=group..'|'..sid..'|'..iid..'|'..epoch..'|'..priority
if persisted and persisted~=payload then
 return reply('REJECT','conflicting_event_receipt') end
-- Locate and check the actual stream event to prevent arbitrary fake receipt.
local rows=redis.call('XRANGE',KEYS[1],sid,sid,'COUNT',1)
if #rows~=1 then
 return reply('REJECT','source_event_missing') end
local values=rows[1][2]
local source={}
for i=1,#values,2 do source[values[i]]=values[i+1] end
if source.event_id~=eid or source.intent_id~=iid or
 source.epoch~=epoch or source.severity~=priority then
 return reply('REJECT','source_event_identity_mismatch') end
if not persisted then
 redis.call('HSET',KEYS[2],eid,payload)
 redis.call('XADD',KEYS[3],'*','event_id',eid,
   'stream_id',sid,'intent_id',iid,'epoch',epoch,
   'severity',priority,'delivery','LOCAL_RECEIPT_ONLY')
end
local ack=redis.call('XACK',KEYS[1],group,sid)
if ack==0 and not persisted then
 -- Unexpected source not pending in group: HSET already happened, but
 -- old Redis Lua has no rollback. Guard with XPENDING preflight below.
 return reply('REJECT','unacknowledged_new_event')
end
return reply('READY',persisted and 'idempotent_receipt' or 'local_receipt_acked')
'''
# Validate group PEL membership before any receipt write. Lua atomicity
# does not provide transactional rollback on a subsequent command failure.
ACK_LUA=ACK_LUA.replace("if not persisted then\n redis.call('HSET'", """if not persisted then
 local pending=redis.call('XPENDING',KEYS[1],group,sid,sid,1)
 if #pending~=1 or pending[1][1]~=sid then
  return reply('REJECT','event_not_pending_for_group') end
 redis.call('HSET'""")

class IsolatedHazardConsumer:
 def __init__(self,r,source,*,group='risk_audit_test_g1',consumer='test_worker_1',
              prefix='research:hazard_delivery:v1'):
  self.r=r;self.source=source
  self.group=group;self.consumer=consumer
  self.receipts=prefix+':receipts'
  self.receipt_stream=prefix+':receipt_stream'
 def setup(self):
  try:self.r.xgroup_create(self.source,self.group,id='0',mkstream=True)
  except Exception as exc:
   if 'BUSYGROUP' not in str(exc):raise
 def poll(self,count=10):
  rows=self.r.xreadgroup(self.group,self.consumer,{self.source:'>'},count=count)
  return [(sid,fields) for _,entries in rows for sid,fields in entries]
 def recover(self,min_idle_ms=0,count=10):
  next_id,rows,_=self.r.xautoclaim(self.source,self.group,self.consumer,
                  min_idle_time=min_idle_ms,start_id='0-0',count=count)
  return rows
 def acknowledge(self,stream_id,fields):
  return json.loads(self.r.eval(ACK_LUA,3,self.source,self.receipts,
       self.receipt_stream,self.group,stream_id,fields.get('event_id',''),
       fields.get('intent_id',''),fields.get('epoch',''),
       fields.get('severity','')))
