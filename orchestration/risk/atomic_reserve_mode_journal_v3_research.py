"""Isolated V3: require generation-bound persistent mode journal in RESERVE Lua.

No production authority: injected risk snapshot AND journal are research data
in isolated Redis. Lua only demonstrates same-transaction checks of risk mode
and exact journal lineage before any atomic RESERVE+PREPARE writes.
"""
from __future__ import annotations
import json
from orchestration.risk.atomic_reserve_prepare_mode_v2_research import LUA as V2

def build_lua():
    s=V2
    before='''local mode=snap.mode
local multiplier=({NORMAL=1.0,RECOVERY=0.70,REDUCED=0.50})[mode]'''
    after='''-- Source mode MUST be bound to a durable mode transition journal.
-- A forged or newer source snapshot cannot silently override REDUCED.
local typmode=redis.call("TYPE",KEYS[9])["ok"]
if typmode~="string" then return answer("REJECT","mode_journal_missing") end
local okmode,mode_state=pcall(cjson.decode,redis.call("GET",KEYS[9]))
if not okmode or type(mode_state)~="table" then
 return answer("REJECT","mode_journal_invalid") end
if not mode_state.epoch or not mode_state.source_gen or
   not mode_state.last_ts or not mode_state.peak or
   not mode_state.mode or mode_state.epoch~=tonumber(snap.mode_epoch) or
   mode_state.source_gen~=snap.mode_generation or
   mode_state.mode~=snap.mode or
   mode_state.last_ts~=tonumber(snap.mode_last_ts_ms) or
   math.abs(tonumber(mode_state.peak)-tonumber(snap.equity_peak))>0.000001 then
 return answer("REJECT","mode_snapshot_generation_or_state_mismatch") end
-- Risk snapshot must not predate the recorded mode transition event.
if tonumber(snap.ts_ms)<tonumber(mode_state.last_ts) then
 return answer("REJECT","snapshot_older_than_mode_journal") end
local mode=snap.mode
local multiplier=({NORMAL=1.0,RECOVERY=0.70,REDUCED=0.50})[mode]'''
    assert s.count(before)==1
    return s.replace(before,after)
LUA=build_lua()
class IsolatedModeJournalReserveV3:
 def __init__(self,r,prefix='research:mode_journal_reserve:v3'):
  self.r=r
  self.keys=[prefix+':gate',prefix+':pending',prefix+':committed',
    prefix+':journal',prefix+':intents',prefix+':clientids',
    prefix+':events',prefix+':risk_snapshot',prefix+':mode_state']
 def op(self,action,*,request_id='',generation='',strategy_id='',
         symbol='',side='',client_order_id='',now_ms=1000000,
         equity=100,original_qty=2,requested_notional=50,proof=''):
  return json.loads(self.r.eval(LUA,len(self.keys),*self.keys,
      action,request_id,generation,strategy_id,symbol,side,
      client_order_id,now_ms,equity,original_qty,requested_notional,proof))
 def test_snapshot(self,*,mode_state,eq,now_ms=1000000,mode=None,
                   generation=None,mode_epoch=None):
  src=dict(source='ISOLATED_TEST_EQUITY_SNAPSHOT',
    mode=mode or mode_state['mode'],equity=eq,available=eq,
    equity_peak=mode_state['peak'],ts_ms=now_ms,health='OK',
    inflight_unknown=False,reservation_authorized=False,
    drawdown_ratio=eq/mode_state['peak']-1,
    mode_generation=generation or mode_state['source_gen'],
    mode_epoch=mode_epoch or mode_state['epoch'],
    mode_last_ts_ms=mode_state['last_ts'])
  self.r.set(self.keys[7],json.dumps(src))
