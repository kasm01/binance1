"""Persistent intent->client-order->fill ledger prototype (RESEARCH ONLY).

Never interfaces with Binance or the production Redis. Single Lua transaction
per journal event; host-side authenticated broker confirmation NOT implemented.
The ledger uses append-only transitions and never silently retries a submitted
or uncertain order. It does not authorize actual execution or release.
"""
import json,re
SCRIPT=r'''
local action=ARGV[1]
local intent=ARGV[2]
local owner=ARGV[3]
local reservation=ARGV[4]
local symbol=ARGV[5]
local side=ARGV[6]
local clientid=ARGV[7]
local ts=tonumber(ARGV[8]) or 0
local orderid=ARGV[9]
local fillid=ARGV[10]
local filled=tonumber(ARGV[11]) or 0
local orig=tonumber(ARGV[12]) or 0
local proof=ARGV[13]
-- Local Redis server boot ID is queried INSIDE Lua so stale AOF OPEN
-- authority is invalid after a new server process starts.
local server_info=redis.call("INFO","server")
local run_id=string.match(server_info,"run_id:([^\r\n]+)")
if not run_id then return cjson.encode({state="REJECT",reason="no_redis_run_id"}) end
local persisted_gate=redis.call("GET",KEYS[1])
local state=(persisted_gate==("OPEN:"..run_id)) and "OPEN" or "CLOSED"
local function ans(s,reason,stage)
 return cjson.encode({state=s,reason=reason,stage=stage or ""})
end
if action=="STATUS" then return ans("READY",state or "CLOSED") end
if action=="FREEZE" then
 redis.call("SET",KEYS[1],"CLOSED")
 return ans("READY","gate_closed")
end
if action=="ENABLE_TEST" then
 if proof~="ISOLATED_OPERATOR_RECONCILED" then return ans("REJECT","reconciliation_required") end
 if redis.call("HLEN",KEYS[2])>0 then return ans("REJECT","pending_intents_need_reconciliation") end
 redis.call("SET",KEYS[1],"OPEN:"..run_id)
 return ans("READY","test_gate_open")
end
if action=="GET" then
 local old=redis.call("HGET",KEYS[2],intent)
 if not old then return ans("UNKNOWN","intent_not_found") end
 return old
end
if state~="OPEN" then return ans("REJECT","gate_closed") end
if intent=="" then return ans("REJECT","missing_intent") end
local raw=redis.call("HGET",KEYS[2],intent)
local rec=raw and cjson.decode(raw) or nil
if action=="PREPARE" then
 if not rec then
  if owner=="" or reservation=="" or clientid=="" or
   (symbol~="NEARUSDT" and symbol~="INJUSDT") or
   (side~="BUY" and side~="SELL") or ts<=0 or orig<=0 then
   return ans("REJECT","invalid_source_identity_or_qty")
  end
  local occupied=redis.call("HGET",KEYS[3],clientid)
  if occupied then return ans("REJECT","client_id_already_owned") end
  rec={intent_id=intent,owner=owner,reservation_id=reservation,
   symbol=symbol,side=side,client_order_id=clientid,
   original_qty=orig,filled_qty=0,stage="PREPARED",created_ms=ts,
   updated_ms=ts,order_id="",fill_ids={}}
  redis.call("HSET",KEYS[2],intent,cjson.encode(rec))
  redis.call("HSET",KEYS[3],clientid,intent)
  return ans("READY","prepared","PREPARED")
 end
 if rec.owner~=owner or rec.reservation_id~=reservation or
   rec.symbol~=symbol or rec.side~=side or rec.client_order_id~=clientid or
   rec.original_qty~=orig then
   return ans("REJECT","immutable_payload_conflict",rec.stage)
 end
 return ans("READY","already_prepared_or_progressed",rec.stage)
end
if not rec then return ans("REJECT","intent_unknown") end
if rec.owner~=owner or rec.reservation_id~=reservation or
   rec.client_order_id~=clientid then
 return ans("REJECT","owner_or_reservation_mismatch",rec.stage)
end
if action=="MARK_SUBMITTING" then
 if rec.stage~="PREPARED" then return ans("REJECT","never_resubmit_unconfirmed",rec.stage) end
 rec.stage="SUBMITTING"
elseif action=="SUBMIT_UNKNOWN" then
 if rec.stage~="SUBMITTING" and rec.stage~="UNKNOWN" then
  return ans("REJECT","invalid_unknown_transition",rec.stage) end
 rec.stage="UNKNOWN"
elseif action=="ACK" then
 if rec.stage~="SUBMITTING" and rec.stage~="UNKNOWN" and rec.stage~="ACKED" then
  return ans("REJECT","invalid_ack_transition",rec.stage) end
 if orderid=="" then return ans("REJECT","missing_exchange_order_id",rec.stage) end
 if rec.order_id~="" and rec.order_id~=orderid then
  return ans("REJECT","conflicting_exchange_order_id",rec.stage) end
 rec.order_id=orderid
 rec.stage="ACKED"
elseif action=="FILL" then
 if rec.stage~="ACKED" and rec.stage~="PARTIAL" and rec.stage~="FILLED" then
  return ans("REJECT","cannot_fill_unconfirmed_order",rec.stage) end
 if rec.order_id=="" or rec.order_id~=orderid or fillid=="" or
    filled<=0 then return ans("REJECT","invalid_fill_reference",rec.stage) end
 for _,x in ipairs(rec.fill_ids) do
   if x.id==fillid then
    if x.qty==filled then return ans("READY","duplicate_fill_ignored",rec.stage) end
    return ans("REJECT","conflicting_fill_duplicate",rec.stage)
   end
 end
 if rec.filled_qty+filled>rec.original_qty+0.00000001 then
  return ans("REJECT","overfill",rec.stage) end
 table.insert(rec.fill_ids,{id=fillid,qty=filled})
 rec.filled_qty=rec.filled_qty+filled
 if math.abs(rec.filled_qty-rec.original_qty)<0.00000001 then
  rec.stage="FILLED"
 else rec.stage="PARTIAL" end
elseif action=="TERMINAL" then
 if rec.stage~="ACKED" and rec.stage~="PARTIAL" then
  return ans("REJECT","invalid_terminal_transition",rec.stage) end
 if rec.order_id=="" or rec.order_id~=orderid then
  return ans("REJECT","terminal_order_mismatch",rec.stage) end
 if proof~="BROKER_TERMINAL_TEST_ONLY" then
  return ans("REJECT","broker_terminal_proof_required",rec.stage) end
 rec.stage="TERMINAL"
elseif action=="FLAT_RECONCILED" then
 if rec.stage~="FILLED" and rec.stage~="TERMINAL" then
  return ans("REJECT","fills_or_terminal_unresolved",rec.stage) end
 if proof~="BROKER_FLAT_TEST_ONLY" then
  return ans("REJECT","broker_flat_proof_required",rec.stage) end
 rec.stage="FLAT_RECONCILED"
else return ans("REJECT","unknown_action",rec.stage)
end
rec.updated_ms=math.max(ts,rec.updated_ms)
redis.call("HSET",KEYS[2],intent,cjson.encode(rec))
redis.call("XADD",KEYS[4],"*","intent_id",intent,"stage",rec.stage,
           "event_time_ms",tostring(ts))
return ans("READY","transition_accepted",rec.stage)
'''
class IsolatedOrderLedger:
 def __init__(self,r,prefix='research:orderledger:v1'):
  self.r=r
  self.keys=[prefix+':gate',prefix+':records',prefix+':clientids',prefix+':events']
 def op(self,action,*,intent='',owner='',reservation='',symbol='',side='',
        client_id='',time_ms=1000,order_id='',fill_id='',fill_qty=0,orig_qty=0,proof=''):
  if action not in ('STATUS','GET','FREEZE','ENABLE_TEST','PREPARE','MARK_SUBMITTING',
     'SUBMIT_UNKNOWN','ACK','FILL','TERMINAL','FLAT_RECONCILED'):
   raise ValueError('unsupported_client_action')
  return json.loads(self.r.eval(SCRIPT,len(self.keys),*self.keys,
       action,intent,owner,reservation,symbol,side,client_id,time_ms,
       order_id,fill_id,fill_qty,orig_qty,proof))
