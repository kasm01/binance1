"""Research-only atomic Redis Lua reservation contract; NEVER live deployment.

The Redis connection MUST refer to an isolated test server, NOT port 6379.
Gate defaults CLOSED on a new/restarted Redis until explicitly reconciled.
"""
import json
SCRIPT=r'''
local action=ARGV[1]
local request_id=ARGV[2]
local strategy=ARGV[3]
local symbol=ARGV[4]
local side=ARGV[5]
local now=tonumber(ARGV[6]) or 0
local equity=tonumber(ARGV[7]) or 0
local token=ARGV[8]
local position=ARGV[9]
local proof=ARGV[10]
local ttl=tonumber(ARGV[11]) or 0
local state=redis.call("GET",KEYS[1])
local function answer(s,reason,notional)
 return cjson.encode({state=s,reason=reason,request_id=request_id,
  allowed_notional=notional or 0})
end
if action=="STATUS" then return answer("READY",state or "CLOSED") end
if action=="CLOSE_GATE" then
 redis.call("SET",KEYS[1],"CLOSED")
 return answer("READY","closed")
end
if action=="OPEN_GATE" then
 if proof~="ISOLATED_TEST_RECONCILED" then return answer("REJECT","reconciliation_required") end
 if redis.call("HLEN",KEYS[2])>0 or redis.call("HLEN",KEYS[3])>0 then
   return answer("REJECT","unresolved_risk_state") end
 redis.call("SET",KEYS[1],"OPEN")
 return answer("READY","test_gate_open")
end
if state~="OPEN" then return answer("REJECT","gate_closed_or_restarted") end
if strategy~="LONG1" and strategy~="SHORT1" then return answer("REJECT","strategy_not_allowed") end
if (strategy=="LONG1" and (symbol~="NEARUSDT" or side~="long")) or
   (strategy=="SHORT1" and (symbol~="INJUSDT" or side~="short")) then
   return answer("REJECT","symbol_side_mismatch")
end
if now<=0 or equity<=0 then return answer("REJECT","invalid_equity_or_time") end
if action=="RESERVE" then
 if request_id=="" or token=="" then return answer("REJECT","missing_ids") end
 local payload=strategy.."|"..symbol.."|"..side.."|"..token
 local cached=redis.call("HGET",KEYS[4],request_id)
 if cached then
  local entry=cjson.decode(cached)
  if entry.payload~=payload then return answer("REJECT","request_id_conflict") end
  return entry.answer
 end
 local response
 if redis.call("HEXISTS",KEYS[2],strategy)==1 or
    redis.call("HEXISTS",KEYS[3],strategy)==1 then
   response=answer("REJECT","strategy_slot_busy")
 else
   local gross=0
   local direction=0
   local count=0
   local symbol_busy=false
   for _,key in ipairs({KEYS[2],KEYS[3]}) do
    local vals=redis.call("HVALS",key)
    for _,v in ipairs(vals) do
     local p=cjson.decode(v)
     local n=tonumber(p.notional)
     gross=gross+n
     count=count+1
     if p.side==side then direction=direction+n end
     if p.symbol==symbol then symbol_busy=true end
    end
   end
   local allowed=math.min(equity*.46,equity-gross,equity*.5-direction)
   if symbol_busy then response=answer("REJECT","symbol_busy")
   elseif count>=2 then response=answer("REJECT","two_slots_full")
   elseif allowed<10 then response=answer("REJECT","budget_below_min")
   else
    local rec={request_id=request_id,reservation_id=token,strategy_id=strategy,
      symbol=symbol,side=side,notional=allowed,expires_at_ms=now+ttl}
    redis.call("HSET",KEYS[2],strategy,cjson.encode(rec))
    response=answer("READY","reserved",allowed)
   end
 end
 redis.call("HSET",KEYS[4],request_id,cjson.encode({payload=payload,answer=response}))
 return response
end
if action=="COMMIT" then
 local raw=redis.call("HGET",KEYS[2],strategy)
 if not raw then
  local committed=redis.call("HGET",KEYS[3],strategy)
  if committed then
   local x=cjson.decode(committed)
   if x.request_id==request_id and x.reservation_id==token and x.position_id==position then
    return answer("READY","already_committed") end
  end
  return answer("REJECT","reservation_not_found")
 end
 local p=cjson.decode(raw)
 if p.request_id~=request_id or p.reservation_id~=token then
  return answer("REJECT","owner_mismatch") end
 if position=="" or now>p.expires_at_ms then
  return answer("REJECT","position_missing_or_expired") end
 p.position_id=position
 redis.call("HSET",KEYS[3],strategy,cjson.encode(p))
 redis.call("HDEL",KEYS[2],strategy)
 return answer("READY","committed")
end
if action=="RELEASE" then
 local raw=redis.call("HGET",KEYS[2],strategy)
 if raw then
  local x=cjson.decode(raw)
  if x.request_id~=request_id or x.reservation_id~=token then
   return answer("REJECT","owner_mismatch") end
  if proof~="EXCHANGE_FLAT_AND_NO_ORDERS" then
   return answer("REJECT","exchange_reconcile_required") end
  redis.call("HDEL",KEYS[2],strategy)
  return answer("READY","released_uncommitted")
 end
 local committed=redis.call("HGET",KEYS[3],strategy)
 if committed then
  local x=cjson.decode(committed)
  if x.request_id~=request_id or x.reservation_id~=token or x.position_id~=position then
   return answer("REJECT","position_owner_mismatch") end
  if proof~="EXCHANGE_FLAT_AND_NO_ORDERS" then
   return answer("REJECT","exchange_reconcile_required") end
  redis.call("HDEL",KEYS[3],strategy)
  return answer("READY","released_committed")
 end
 return answer("READY","already_released")
end
return answer("REJECT","unknown_action")
'''
class IsolatedRedisRiskV2:
 def __init__(self,client,prefix="research:central:risk:v2"):
  self.client=client
  self.keys=[prefix+":gate",prefix+":pending",prefix+":committed",prefix+":journal"]
 def act(self,action,request_id="",strategy_id="",symbol="",side="",
         now_ms=1000,equity=100,reservation_id="",position_id="",
         verified="",ttl_ms=60000):
  return json.loads(self.client.eval(SCRIPT,len(self.keys),*self.keys,
    action,request_id,strategy_id,symbol,side,now_ms,equity,
    reservation_id,position_id,verified,ttl_ms))
