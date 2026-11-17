"""Durable central DD mode transition journal. ISOLATED RESEARCH ONLY.

Models actual enforce-dry profile 12% enter REDUCED, 6% enter RECOVERY,
two profitable closes opened in RECOVERY to return NORMAL. Fail closed on
stale generation, timestamp, event ID, malformed inputs or Redis restart.
Events are synthetic; no authenticated Binance authority in this module.
"""
from __future__ import annotations
import json
LUA=r'''
-- KEYS: state, idempotency, events, run_gate
-- ARGV: command,event_id,expected_epoch,source_time_ms,equity,pnl,
--       opened_mode,risk_stop,source_generation
local action,eid,opened,riskstop,source_gen=ARGV[1],ARGV[2],ARGV[7],ARGV[8],ARGV[9]
local expected=tonumber(ARGV[3])
local ts=tonumber(ARGV[4])
local eq=tonumber(ARGV[5])
local pnl=tonumber(ARGV[6])
local function answer(s,reason,epoch,mode)
 return cjson.encode({state=s,reason=reason,epoch=epoch or 0,
 mode=mode or 'UNKNOWN',production_authorized=false})
end
local rid=string.match(redis.call('INFO','server'),'run_id:([^\r\n]+)')
if not rid then return answer('REJECT','no_server_run_id') end
if action=='GET' then
 local raw=redis.call('GET',KEYS[1])
 if not raw then return answer('UNKNOWN','state_missing') end
 local x=cjson.decode(raw)
 return answer('READY','stored',x.epoch,x.mode)
end
if action=='CLOSE' then
 redis.call('SET',KEYS[4],'CLOSED')
 return answer('READY','gate_closed')
end
if action=='INIT_TEST' then
 if redis.call('EXISTS',KEYS[1])==1 then return answer('REJECT','already_initialized') end
 -- INIT_TEST requires prior peak equity, passed in pnl argument only for
 -- isolated test. Do not infer past peak from a single broker snapshot.
 local initpeak=pnl
 if source_gen=='' or not eq or eq<=0 or not ts or ts<=0
 or eq~=eq or not initpeak or initpeak<eq or initpeak~=initpeak
 then return answer('REJECT','invalid_init_or_missing_historical_peak') end
 for _,pair in ipairs({{KEYS[1],'string'},{KEYS[2],'hash'},
  {KEYS[3],'stream'},{KEYS[4],'string'}}) do
  local typ=redis.call('TYPE',pair[1])['ok']
  if typ~='none' and typ~=pair[2] then
   return answer('REJECT','bad_redis_storage_type') end
 end
 local state={epoch=1,mode='REDUCED',peak=initpeak,recovery_successes=0,
  last_ts=ts,source_gen=source_gen,drawdown_ratio=eq/initpeak-1}
 redis.call('SET',KEYS[1],cjson.encode(state))
 redis.call('SET',KEYS[4],'CLOSED')
 redis.call('XADD',KEYS[3],'*','event_id','INIT','epoch','1',
             'mode','REDUCED','source_time_ms',tostring(ts))
 return answer('READY','research_initial_reduced',1,'REDUCED')
end
if action~='MTM' and action~='CLOSE_EVENT' then
 return answer('REJECT','unsupported_action') end
if not expected or not ts or not eq or not pnl or
 eq<=0 or eq~=eq or pnl~=pnl or expected<1 or ts<=0 or
 eid=='' or source_gen=='' then return answer('REJECT','invalid_input') end
local raw=redis.call('GET',KEYS[1])
if not raw then return answer('REJECT','mode_state_missing') end
local x=cjson.decode(raw)
if x.source_gen~=source_gen then return answer('REJECT','source_generation_mismatch',x.epoch,x.mode) end
if expected~=x.epoch then return answer('REJECT','stale_epoch',x.epoch,x.mode) end
if ts<=x.last_ts then return answer('REJECT','stale_source_time',x.epoch,x.mode) end
if redis.call('HEXISTS',KEYS[2],eid)==1 then
 return answer('REJECT','event_id_already_seen',x.epoch,x.mode) end
local last=x.mode
local mode=x.mode
local peak=math.max(x.peak,eq)
local dd=eq/peak-1
local wins=x.recovery_successes
if mode=='NORMAL' and dd<=-0.12 then mode='REDUCED'; wins=0
elseif mode=='REDUCED' and dd>=-0.06 then mode='RECOVERY';wins=0
elseif mode=='RECOVERY' then
 if dd<=-0.12 or (action=='CLOSE_EVENT' and riskstop=='true') then
  mode='REDUCED';wins=0
 elseif action=='CLOSE_EVENT' and opened=='RECOVERY' and pnl>0 then
  wins=wins+1
  if wins>=2 then mode='NORMAL';wins=0 end
 end
end
for _,pair in ipairs({{KEYS[1],'string'},{KEYS[2],'hash'},
 {KEYS[3],'stream'},{KEYS[4],'string'}}) do
 local typ=redis.call('TYPE',pair[1])['ok']
 if typ~='none' and typ~=pair[2] then
  return answer('REJECT','bad_redis_storage_type',x.epoch,x.mode) end
end
x.epoch=x.epoch+1;x.mode=mode;x.peak=peak
x.recovery_successes=wins;x.drawdown_ratio=dd;x.last_ts=ts
redis.call('SET',KEYS[1],cjson.encode(x))
redis.call('SET',KEYS[4],'CLOSED')
redis.call('HSET',KEYS[2],eid,tostring(x.epoch))
redis.call('XADD',KEYS[3],'*','event_id',eid,'epoch',tostring(x.epoch),
 'from_mode',last,'mode',mode,'source_time_ms',tostring(ts),
 'drawdown',tostring(dd),'recovery_successes',tostring(wins))
return answer('READY','mode_transition_recorded',x.epoch,x.mode)
'''
class IsolatedRiskModeJournal:
 def __init__(self,r,prefix='research:mode_journal:v1'):
  self.r=r
  self.keys=[prefix+':state',prefix+':seen',prefix+':events',prefix+':gate']
 def act(self,action,*,event_id='',expected_epoch=0,source_time_ms=0,
   equity=0,pnl=0,opened_risk_mode='',is_risk_stop=False,
   generation='test-generation'):
  return json.loads(self.r.eval(LUA,4,*self.keys,action,event_id,expected_epoch,
   source_time_ms,equity,pnl,opened_risk_mode,
   'true' if is_risk_stop else 'false',generation))
 def state(self):
  raw=self.r.get(self.keys[0])
  return json.loads(raw) if raw else None
