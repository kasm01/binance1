"""AST extract actual SHORT1 entry and exchange helper; no network/Redis."""
import ast,copy,json
from pathlib import Path
from core.joint_risk_live_guard_v1 import check_new_open
tree=ast.parse(Path("core/trade_executor_short.py").read_text())
classdef=next(c for c in ast.walk(tree) if isinstance(c,ast.ClassDef) and
 any(getattr(m,"name",None)=="open_position_from_signal" for m in c.body))
names={"open_position_from_signal","_exchange_open_market"}
nodes=[copy.deepcopy(n) for n in classdef.body if getattr(n,"name",None) in names]
assert {n.name for n in nodes}==names
for n in nodes:
 n.decorator_list=[];n.returns=None
 for a in n.args.args+n.args.kwonlyargs:a.annotation=None
module=ast.Module(body=[ast.ClassDef(name="FakeShort",bases=[],keywords=[],body=nodes,decorator_list=[])],type_ignores=[])
ast.fix_missing_locations(module)
ns={"Optional":__import__("typing").Optional}
exec(compile(module,"<actual-short-entry-ast>","exec"),ns)
S=ns["FakeShort"]
class Client:
 def __init__(self):self.order_calls=0
 def futures_create_order(self,**kwargs):
  self.order_calls+=1
  raise AssertionError("SHOULD NEVER ORDER")
x=S();x.client=Client();x.dry_run=False;x.strategy_id="SHORT1"
res=x.open_position_from_signal("INJUSDT","short","1m",{})
assert res["status"]=="skip" and res["reason"]=="joint_risk_live_authority_not_connected",res
assert x.client.order_calls==0
try:
 x._exchange_open_market("INJUSDT",1)
except RuntimeError as err:
 assert "joint_risk_live_authority_not_connected" in str(err)
else:raise AssertionError("direct live order open would reach exchange")
assert x.client.order_calls==0
# DRY_RUN is still eligible for existing short shadow central client flow;
# direct market emitter never submits a real order.
x.dry_run=True
shadow=x._exchange_open_market("INJUSDT",1)
assert shadow["status"]=="dry_run" and x.client.order_calls==0
assert check_new_open(strategy_id="SHORT1",symbol="INJUSDT",side="short",dry_run=False).allow is False
assert check_new_open(strategy_id="LONG1",symbol="NEARUSDT",side="long",dry_run=False).allow is False
print(json.dumps({"actual_short_live_entry_skip":True,
 "actual_short_direct_exchange_open_blocked":True,
 "short_dry_run_exchange_helper_unchanged":True,
 "identical_shared_live_gate":True,
 "binance_orders":0,"redis_writes":0},sort_keys=True))
