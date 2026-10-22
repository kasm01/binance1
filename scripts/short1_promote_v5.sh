#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/kasm920/venv/bin/python
GATE=research/short/results/SHORT1_V30_LIVE_PROMOTION_GATE.json
SRC_MODEL=research/short/models/short_model_INJUSDT_3m_fresh_v5.joblib
SRC_META=research/short/models/short_model_meta_INJUSDT_3m_fresh_v5.json
DST_MODEL=models/short_model_INJUSDT_3m_best.joblib
DST_META=models/short_model_meta_INJUSDT_3m.json
[[ -f "$GATE" && -f "$SRC_MODEL" && -f "$SRC_META" ]] || { echo '[FAIL] required gate/model/meta missing'; exit 2; }
# Load only safety keys from .env.
ARMED_VAL="$(grep -E '^ARMED=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
KILL_VAL="$(grep -E '^LIVE_KILL_SWITCH=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
[[ "$ARMED_VAL" == "0" && "$KILL_VAL" == "1" ]] || { echo "[FAIL] system must be DISARMED (ARMED=0 KILL=1), got ARMED=$ARMED_VAL KILL=$KILL_VAL"; exit 3; }
STATUS="$($PY - <<'PY'
import json
p=json.load(open('research/short/results/SHORT1_V30_LIVE_PROMOTION_GATE.json'))
print(p.get('promotion_status',''))
PY
)"
[[ "$STATUS" == "PRODUCTION_ELIGIBLE" ]] || { echo "[FAIL] gate=$STATUS; promotion refused"; exit 4; }
mkdir -p models research/short/promotion_backups
TS="$(date -u +%Y%m%d_%H%M%S)"
[[ ! -f "$DST_MODEL" ]] || cp "$DST_MODEL" "research/short/promotion_backups/short_model_INJUSDT_3m_best.joblib.$TS"
[[ ! -f "$DST_META" ]] || cp "$DST_META" "research/short/promotion_backups/short_model_meta_INJUSDT_3m.json.$TS"
cp "$SRC_MODEL" "$DST_MODEL.tmp"
$PY - <<'PY'
import json
from pathlib import Path
src=Path('research/short/models/short_model_meta_INJUSDT_3m_fresh_v5.json')
dst=Path('models/short_model_meta_INJUSDT_3m.json.tmp')
o=json.loads(src.read_text())
o['promoted']=True; o['status']='production'; o['promotion_source']='SHORT1_V30_LIVE_PROMOTION_GATE.json'
dst.write_text(json.dumps(o,indent=2,allow_nan=False))
PY
mv "$DST_MODEL.tmp" "$DST_MODEL"
mv "$DST_META.tmp" "$DST_META"
$PY - <<'PY'
from models.short_model_registry import ShortModelRegistry
m=ShortModelRegistry().get('INJUSDT','3m',require_promoted=True,require_execution_contract=True)
print('PROMOTED_MODEL_OK',m.meta.get('provider'),m.meta.get('status'),m.meta.get('promoted'))
PY
echo '[OK] SHORT1 V5 promoted. Trade services remain untouched.'
