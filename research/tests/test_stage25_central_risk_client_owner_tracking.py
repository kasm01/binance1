"""Actual SHORT1 CentralRiskClient owner tracking without network/Redis."""
from __future__ import annotations
import json
from orchestration.risk.central_risk_client import CentralRiskClient

def client():
 x=CentralRiskClient.__new__(CentralRiskClient)
 x._dry_reservation_owner_by_strategy={}
 x.sent=[]
 def fake_ask(request):
  x.sent.append(request)
  action=request["action"]
  if action=="EVALUATE_OPEN":
   return {"state":"READY","reason":"central_risk_decision",
           "request_id":"central-owned-A","decision":{
              "production_ready":False,"allowed_notional":15}}
  if action=="RELEASE":
   return {"state":"READY","reason":"reservation_released_by_owner"}
  raise AssertionError(action)
 x._ask=fake_ask
 return x
x=client()
assert x.release(strategy_id="SHORT1")["reason"]=="reservation_owner_unknown"
assert x.sent==[]
a=x.evaluate_open(strategy_id="SHORT1",symbol="INJUSDT",side="short",
                  market_context_multiplier=1,reserve=True)
assert a["state"]=="READY"
assert x._dry_reservation_owner_by_strategy["SHORT1"]=="central-owned-A"
released=x.release(strategy_id="SHORT1")
assert released["reason"]=="reservation_released_by_owner"
assert x.sent[-1]["reservation_owner_request_id"]=="central-owned-A"
assert x.release(strategy_id="SHORT1")["state"]=="REJECT"
assert "SHORT1" not in x._dry_reservation_owner_by_strategy
# READY without actual reservation MUST NOT mint a release owner.
y=client()
assert y.evaluate_open(strategy_id="SHORT1",symbol="INJUSDT",side="short",
             market_context_multiplier=1,reserve=False)["state"]=="READY"
assert y.release(strategy_id="SHORT1")["reason"]=="reservation_owner_unknown"
# A newer client instance lacks old ownership: explicit fail-closed.
z=client()
assert z.release(strategy_id="SHORT1")["state"]=="REJECT"
# Failed release must RETAIN owner and be retryable.
f=client()
f.evaluate_open(strategy_id="SHORT1",symbol="INJUSDT",side="short",
                market_context_multiplier=1,reserve=True)
f._ask=lambda req:{"state":"REJECT","reason":"redis_unavailable"}
assert f.release(strategy_id="SHORT1")["state"]=="REJECT"
assert f._dry_reservation_owner_by_strategy["SHORT1"]=="central-owned-A"
print(json.dumps({"actual_short_client_owner_tracking":True,
 "missing_owner_denied":True,"correct_owner_transmitted":True,
 "owner_cleared_only_on_success":True,
 "dry_no_reserve_no_owner":True,"restart_without_owner_denied":True,
 "failed_release_owner_retained":True,
 "external_requests":0,"real_redis_writes":0},sort_keys=True))
