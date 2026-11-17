"""480 actually received V6 events, original event-time replay, NO Redis writes."""
import json,redis,csv
from collections import Counter
from pathlib import Path
from orchestration.shadow.short1_v6_master_contract_v1 import audit
r=redis.Redis(decode_responses=True,socket_timeout=2)
items=r.xrevrange("telemetry:short1:inj:model_shadow_v6",count=480)
out=[]
for sid,data in reversed(items):
 record=json.loads(data["json"])
 ms=int(record["completed_close_time"])
 result=audit(record,now_ms=ms+5000)
 result["source_redis_id"]=sid
 out.append(result)
p=Path("research/results/SHORT1_INJ_V6_MASTER_CONTRACT_480_20261113.csv")
with p.open("w",newline="") as f:
    fields=["source_redis_id","source_event_id","completed_close_ms","state","reason",
            "v6_score","v6_threshold","missing","can_authorize_executor",
            "flow15","flow60","price_return15","oi_available"]
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
    writer.writeheader()
    for x in out:
        row=dict(x)
        if isinstance(row.get("missing"),list):row["missing"]=",".join(row["missing"])
        writer.writerow(row)
counts=Counter((x["state"],x["reason"]) for x in out)
print("AUDITED_EVENTS",len(out),"DECISIONS",dict(counts))
print("V6_ABOVE_OWN_THRESHOLD",sum(x.get("v6_score") is not None for x in out))
print("AUTHORISED_EXECUTOR",sum(x["can_authorize_executor"] for x in out))
print("RESULT",p)
