"""LONG1+SHORT1 shared RESERVE + PREPARE atomically, RESEARCH ONLY.

This is a separate isolated Redis experiment, NOT production logic. The risk
budget is notional exposure, not actual Binance margin. No auth, exchange
account/position reconciliation, stop attachment or real execution.
Lua script never emits a Binance order or permission to place one.
"""
from __future__ import annotations
import json

LUA=r'''
-- KEYS: gate,pending,committed,reservation_journal,intents,client_ids,events
-- ARGV: action,request,gen,strategy,symbol,side,client_id,now_ms,equity,
--       orig_qty,notional_requested,proof
local action=ARGV[1]
local req=ARGV[2]
local gen=ARGV[3]
local strat=ARGV[4]
local sym=ARGV[5]
local side=ARGV[6]
local clientid=ARGV[7]
local now=tonumber(ARGV[8])
local equity=tonumber(ARGV[9])
local qty=tonumber(ARGV[10])
local asked=tonumber(ARGV[11])
local proof=ARGV[12]
local rid=string.match(redis.call("INFO","server"),"run_id:([^\r\n]+)")
if not rid then return cjson.encode({state="REJECT",reason="redis_run_id_unavailable"}) end
local open=redis.call("GET",KEYS[1])==("OPEN:"..rid)
local function answer(state,reason,allowed)
 return cjson.encode({state=state,reason=reason,allowed_notional=allowed or 0,
   order_submission_authorized=false})
end
if action=="STATUS" then return answer("READY",open and "OPEN" or "CLOSED") end
if action=="CLOSE" then
 redis.call("SET",KEYS[1],"CLOSED")
 return answer("READY","closed")
end
if action=="OPEN_TEST" then
 if proof~="ISOLATED_TEST_RECONCILED" then return answer("REJECT","reconciliation_required") end
 if redis.call("HLEN",KEYS[2])>0 or redis.call("HLEN",KEYS[3])>0
  or redis.call("HLEN",KEYS[5])>0 then
  return answer("REJECT","unresolved_intent_or_exposure")
 end
 redis.call("SET",KEYS[1],"OPEN:"..rid)
 return answer("READY","test_open")
end
if not open then return answer("REJECT","gate_closed_or_restart") end
if action~="RESERVE_PREPARE" then return answer("REJECT","unsupported_action") end
-- Critical: Redis Lua scripts are atomic to other clients but not a rollback
-- transaction. Wrong-type XADD/HSET at the end could leave partial records.
-- Reject incompatible key types BEFORE first mutation.
for _,k in ipairs({KEYS[2],KEYS[3],KEYS[4],KEYS[5],KEYS[6]}) do
 local typ=redis.call("TYPE",k)["ok"]
 if typ~="none" and typ~="hash" then return answer("REJECT","risk_store_wrong_type") end
end
local typ=redis.call("TYPE",KEYS[7])["ok"]
if typ~="none" and typ~="stream" then return answer("REJECT","audit_event_stream_wrong_type") end
if req=="" or gen=="" or clientid=="" then return answer("REJECT","missing_request_identity") end
if (strat~="LONG1" or sym~="NEARUSDT" or side~="long") and
   (strat~="SHORT1" or sym~="INJUSDT" or side~="short") then
 return answer("REJECT","strategy_symbol_side_invalid")
end
if not now or not equity or not qty or not asked or now<=0 or equity<=0 or
   qty<=0 or asked<=0 or asked~=asked or qty~=qty or equity~=equity then
 return answer("REJECT","invalid_number")
end
-- Bind all material request properties; even the requested notional cannot
-- mutate during idempotent replay. This journal must remain durable.
local payload=gen.."|"..strat.."|"..sym.."|"..side.."|"..clientid.."|"..
              tostring(qty).."|"..tostring(asked)
local journal=redis.call("HGET",KEYS[4],req)
if journal then
 local j=cjson.decode(journal)
 if j.payload~=payload then return answer("REJECT","request_id_payload_conflict") end
 return j.answer
end
if redis.call("HEXISTS",KEYS[2],strat)==1 or
   redis.call("HEXISTS",KEYS[3],strat)==1 then
 return answer("REJECT","strategy_slot_occupied")
end
if redis.call("HEXISTS",KEYS[5],req)==1 then
 return answer("REJECT","intent_id_occupied_without_journal")
end
if redis.call("HEXISTS",KEYS[6],clientid)==1 then
 return answer("REJECT","client_order_id_occupied")
end
local gross=0
local side_used=0
local count=0
for _,key in ipairs({KEYS[2],KEYS[3]}) do
 local rows=redis.call("HVALS",key)
 for _,raw in ipairs(rows) do
  local row=cjson.decode(raw)
  local notional=tonumber(row.notional)
  if not notional or notional<0 or notional~=notional then
   return answer("REJECT","corrupted_existing_reservation")
  end
  count=count+1
  gross=gross+notional
  if row.side==side then side_used=side_used+notional end
  if row.symbol==sym then return answer("REJECT","symbol_slot_occupied") end
 end
end
if count>=2 then return answer("REJECT","max_two_positions") end
-- Notional upper bound (not account margin); never promote client request
-- into a larger allocation than the central policy, and cap cross-bot gross.
local allowed=math.min(asked,equity*.46,equity-gross,equity*.5-side_used)
if allowed<10 then return answer("REJECT","insufficient_headroom") end
local reserve={request_id=req,reservation_id=gen,strategy_id=strat,
  symbol=sym,side=side,notional=allowed,created_ms=now,
  status="RESERVED_PREPARED"}
local intent={intent_id=req,reservation_id=gen,owner=strat,
  symbol=sym,side=side=="long" and "BUY" or "SELL",
  client_order_id=clientid,original_qty=qty,filled_qty=0,
  stage="PREPARED",exchange_order_id="",created_ms=now}
local reply=answer("READY","atomic_reserved_prepared",allowed)
-- Lua atomicity: both records and ownership index are written within one
-- transaction, along with the request journal and audit event.
redis.call("HSET",KEYS[2],strat,cjson.encode(reserve))
redis.call("HSET",KEYS[5],req,cjson.encode(intent))
redis.call("HSET",KEYS[6],clientid,req)
redis.call("HSET",KEYS[4],req,cjson.encode({payload=payload,answer=reply}))
redis.call("XADD",KEYS[7],"*","request_id",req,"state","RESERVED_PREPARED")
return reply
'''
class IsolatedAtomicReservePrepare:
    def __init__(self,r,prefix='research:atomic_reserve_prepare:v1'):
        self.r=r
        self.keys=[prefix+':gate',prefix+':pending',prefix+':committed',
                   prefix+':reservation_journal',prefix+':intents',
                   prefix+':client_ids',prefix+':events']
    def act(self,action,*,request_id='',generation='',strategy_id='',
            symbol='',side='',client_order_id='',now_ms=1000,equity=100.0,
            original_qty=3,requested_notional=46,proof=''):
        return json.loads(self.r.eval(LUA,len(self.keys),*self.keys,
            action,request_id,generation,strategy_id,symbol,side,
            client_order_id,now_ms,equity,original_qty,requested_notional,proof))
