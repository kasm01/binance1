"""Isolated DRY-RUN reconciler: bind immutable Redis risk+intent to broker evidence.

Research-only. No Binance calls or production Redis connection are made here.
A broker adapter may be injected, but it only issues read-only requests.
The audit CAS is atomic, idempotent and fingerprinted; it is NOT a real
authorization credential. All decisions prohibit trade, COMMIT and RELEASE.
"""
from __future__ import annotations
import hashlib,json,math,time
from orchestration.risk.bound_exchange_reconcile_v1 import bind

# Gate and state are stored by the atomic research reservation module.
AUDIT_LUA=r'''
-- KEYS risk_pending, ledger_intents, event_audit, gate, event_hash
-- ARGV intent_id, generation, expected_client_id, evidence_digest, label,
--     observed_at, broker_elapsed
local iid,gen,cid,digest,label=ARGV[1],ARGV[2],ARGV[3],ARGV[4],ARGV[5]
local observed=tonumber(ARGV[6]);local elapsed=tonumber(ARGV[7])
local function out(state,why) return cjson.encode({state=state,reason=why}) end
if not observed or observed<=0 or not elapsed or elapsed<0 or elapsed>15 then
 return out('REJECT','invalid_observation_timestamp')
end
if label~='UNKNOWN_FREEZE' and label~='RECOVERED_FILL_PENDING_JOURNAL'
 and label~='HOLD_RECONCILIATION' and label~='PROTECTED_EVIDENCE_ONLY'
 and label~='FREEZE_UNPROTECTED' and label~='FLAT_EVIDENCE_ONLY' then
 return out('REJECT','unexpected_evidence_state')
end
local existing=redis.call('HGET',KEYS[5],iid)
if existing then
 local x=cjson.decode(existing)
 if x.gen==gen and x.clientid==cid and x.digest==digest and x.label==label then
  return out('READY','idempotent_audit_replay')
 end
 -- conflicting evidence after one audit requires a new monotonic epoch,
 -- not silent overwrite or risk promotion.
 return out('REJECT','conflicting_evidence_needs_new_epoch')
end
local raw=redis.call('HGET',KEYS[2],iid)
if not raw then return out('REJECT','intent_not_found') end
local intent=cjson.decode(raw)
if intent.intent_id~=iid or intent.reservation_id~=gen
  or intent.client_order_id~=cid then
 return out('REJECT','ledger_owner_mismatch')
end
local reserve=redis.call('HGET',KEYS[1],intent.owner)
if not reserve then return out('REJECT','risk_reservation_not_found') end
local r=cjson.decode(reserve)
if r.request_id~=iid or r.reservation_id~=gen or r.symbol~=intent.symbol
 or r.strategy_id~=intent.owner then
 return out('REJECT','reservation_generation_mismatch')
end
for _,pair in ipairs({{KEYS[5],'hash'},{KEYS[3],'stream'}}) do
 local t=redis.call('TYPE',pair[1])['ok']
 if t~='none' and t~=pair[2] then return out('REJECT','invalid_audit_storage_type') end
end
-- Fail CLOSED for all states including successful broker observations.
redis.call('SET',KEYS[4],'CLOSED')
redis.call('HSET',KEYS[5],iid,cjson.encode({gen=gen,clientid=cid,
 digest=digest,label=label,observed_at=observed,elapsed=elapsed}))
redis.call('XADD',KEYS[3],'*','intent_id',iid,'state',label,
 'observed_at',tostring(observed),'digest',digest)
return out('READY','audit_persisted_gate_closed')
'''

class IsolatedReadOnlyReconcileWorker:
    def __init__(self,redis_client,reserve_keys,broker,*,prefix='research:reconcile:v1',
                 clock=time.monotonic,utc_ms=lambda:int(time.time()*1000)):
        if len(reserve_keys)!=7:raise ValueError('invalid_isolated_risk_keys')
        self.redis=redis_client
        self.reserve_keys=reserve_keys
        self.broker=broker
        self.clock=clock
        self.utc_ms=utc_ms
        self.audit_keys=[prefix+':events',prefix+':hash']
    def inspect(self,intent_id):
        """Pure read from an isolated Redis; broker is read-only adapter."""
        raw=self.redis.hget(self.reserve_keys[4],intent_id)
        if not raw:
            return dict(state='UNKNOWN_FREEZE',reason='intent_not_found',can_submit=False,
                        can_commit=False,can_release=False)
        ledger=json.loads(raw)
        owner=ledger.get('owner')
        reservation_raw=self.redis.hget(self.reserve_keys[1],owner)
        if not reservation_raw:
            return dict(state='UNKNOWN_FREEZE',reason='reservation_not_found',can_submit=False,
                        can_commit=False,can_release=False)
        reservation=json.loads(reservation_raw)
        return bind(reservation=reservation,ledger=ledger,
            read_order=self.broker.get_order,read_fills=self.broker.get_fills,
            read_positions=self.broker.get_positions,
            read_open_orders=self.broker.get_open_orders,
            read_open_algo=self.broker.get_open_algos,
            clock=self.clock)
    def inspect_and_persist(self,intent_id):
        """Research CAS only. A reviewable audit is never execution permission."""
        started=self.clock()
        try:
            result=self.inspect(intent_id)
            raw=self.redis.hget(self.reserve_keys[4],intent_id)
            if not raw:
                return dict(state='REJECT',reason='ledger_missing_after_inspection')
            rec=json.loads(raw)
            # Keep a bounded evidence fingerprint without writing client IDs
            # or exchange secrets to the audit stream.
            digest=hashlib.sha256(json.dumps(result,sort_keys=True,
                    separators=(',',':')).encode()).hexdigest()
            elapsed=self.clock()-started
            if not math.isfinite(elapsed) or elapsed>15:
                result=dict(state='UNKNOWN_FREEZE',reason='worker_or_network_too_slow')
            audit=json.loads(self.redis.eval(AUDIT_LUA,5,
               self.reserve_keys[1],self.reserve_keys[4],self.audit_keys[0],
               self.reserve_keys[0],self.audit_keys[1],
               intent_id,rec['reservation_id'],rec['client_order_id'],digest,
               result['state'],self.utc_ms(),elapsed))
            return dict(audit=audit,evidence=result,
                        can_submit=False,can_commit=False,can_release=False)
        except Exception as exc:
            # No silent authorization, even if Redis or broker is unavailable.
            return dict(state='UNKNOWN_FREEZE',reason='worker_failed:'+type(exc).__name__,
                        can_submit=False,can_commit=False,can_release=False)
