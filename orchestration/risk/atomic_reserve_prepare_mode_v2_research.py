"""Isolated research V2: atomically validate stored equity/DD mode and reserve.

DERIVED from the previously tested V1 Lua: no change to running services.
Risk snapshot is test-injected into isolated Redis; not exchange-authenticated.
No Binance order permission. Snapshot budget is an illustrative notional cap.
"""
from __future__ import annotations
import json
from orchestration.risk.atomic_reserve_prepare_v1_research import LUA as BASE_LUA

def build_lua():
    s=BASE_LUA
    old='''local proof=ARGV[12]
local rid='''
    new='''local proof=ARGV[12]
-- Test-only risk snapshot stored at KEYS[8], never supplied as a mode flag
-- by the order caller. Actual production will require authenticated source.
local risk_snapshot=redis.call("GET",KEYS[8])
local rid='''
    assert s.count(old)==1
    s=s.replace(old,new)
    old='''if req=="" or gen=="" or clientid=="" then return answer("REJECT","missing_request_identity") end'''
    new='''if req=="" or gen=="" or clientid=="" then return answer("REJECT","missing_request_identity") end
local snap_typ=redis.call("TYPE",KEYS[8])["ok"]
if snap_typ~="string" then return answer("REJECT","risk_snapshot_missing") end
local ok,snap=pcall(cjson.decode,risk_snapshot)
if not ok or type(snap)~="table" then return answer("REJECT","risk_snapshot_invalid_json") end
if snap.source~="ISOLATED_TEST_EQUITY_SNAPSHOT" or
   snap.health~="OK" or snap.inflight_unknown~=false or
   snap.reservation_authorized~=false then
 return answer("REJECT","risk_snapshot_untrusted_or_unhealthy") end
local mode=snap.mode
local multiplier=({NORMAL=1.0,RECOVERY=0.70,REDUCED=0.50})[mode]
if not multiplier then return answer("REJECT","risk_mode_invalid") end
local snap_ts=tonumber(snap.ts_ms)
local snap_equity=tonumber(snap.equity)
local available=tonumber(snap.available)
local peak=tonumber(snap.equity_peak)
local max_age=120000
if not snap_ts or not snap_equity or not available or not peak or
   snap_equity<=0 or available<0 or peak<snap_equity or
   snap_equity~=snap_equity or peak~=peak or available~=available or
   snap_ts>now+2000 or now-snap_ts>max_age then
 return answer("REJECT","risk_snapshot_invalid_or_stale") end
if not equity or math.abs(equity-snap_equity)>0.000001 then
 return answer("REJECT","caller_equity_disagrees_with_snapshot") end
local dd=snap_equity/peak-1
local stated_dd=tonumber(snap.drawdown_ratio)
if not stated_dd or stated_dd~=stated_dd or
   math.abs(dd-stated_dd)>0.003 then
 return answer("REJECT","risk_snapshot_drawdown_inconsistent") end
-- Fail-closed when reported state is less cautious than the known policy.
if mode=="NORMAL" and dd<=-0.12 then
 return answer("REJECT","normal_mode_inconsistent_with_drawdown") end
if mode=="RECOVERY" and dd<=-0.12 then
 return answer("REJECT","recovery_mode_inconsistent_with_drawdown") end
'''
    assert s.count(old)==1
    s=s.replace(old,new)
    old='''local payload=gen.."|"..strat.."|"..sym.."|"..side.."|"..clientid.."|"..
              tostring(qty).."|"..tostring(asked)'''
    new='''local payload=gen.."|"..strat.."|"..sym.."|"..side.."|"..clientid.."|"..
              tostring(qty).."|"..tostring(asked)'''
    # Existing same-request idempotent replies remain replayable only while
    # source gate is OPEN; the budget cannot change on replay after a mode
    # update. A new request must be used following a mode change.
    assert s.count(old)==1
    old='''local allowed=math.min(asked,equity*.46,equity-gross,equity*.5-side_used)'''
    new='''-- Same profile as central_risk_enforce_dry:
-- requested_notional * .92 operational headroom * mode multiplier
local policy=asked*0.92*multiplier
local allowed=math.min(policy,equity-gross,equity*.5-side_used,
                       equity*.5)'''
    assert s.count(old)==1
    s=s.replace(old,new)
    old='''  symbol=sym,side=side,notional=allowed,created_ms=now,
  status="RESERVED_PREPARED"}'''
    new='''  symbol=sym,side=side,notional=allowed,created_ms=now,
  risk_mode=mode,risk_snapshot_ts_ms=snap_ts,
  status="RESERVED_PREPARED"}'''
    assert s.count(old)==1
    s=s.replace(old,new)
    # IMPORTANT avoid using untrusted snapshot state as a permission to send.
    return s

LUA=build_lua()
class IsolatedModeRiskReserve:
    def __init__(self,r,prefix='research:atomic_mode_reserve:v2'):
        self.r=r
        self.keys=[prefix+':gate',prefix+':pending',prefix+':committed',
                   prefix+':journal',prefix+':intents',prefix+':clientids',
                   prefix+':events',prefix+':risk_snapshot']
    def op(self,action,*,request_id='',generation='',strategy_id='',
           symbol='',side='',client_order_id='',now_ms=1000000,equity=100,
           original_qty=2,requested_notional=50,proof=''):
        return json.loads(self.r.eval(LUA,8,*self.keys,
                         action,request_id,generation,strategy_id,
                         symbol,side,client_order_id,now_ms,equity,
                         original_qty,requested_notional,proof))
    def test_snapshot(self,*,mode,equity=100,available=100,peak=100,
                      ts_ms=1000000,health='OK',inflight_unknown=False,
                      drawdown_ratio=None):
        # Test fixture only, NEVER use to seed real exchange data.
        val=dict(source='ISOLATED_TEST_EQUITY_SNAPSHOT',mode=mode,
                 equity=equity,available=available,equity_peak=peak,
                 ts_ms=ts_ms,health=health,inflight_unknown=inflight_unknown,
                 reservation_authorized=False,drawdown_ratio=(
                     equity/peak-1 if drawdown_ratio is None else drawdown_ratio))
        self.r.set(self.keys[7],json.dumps(val))
