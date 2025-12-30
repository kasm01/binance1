#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/binance1}"
cd "$PROJECT_DIR"

SYMBOL="${SYMBOL:-BTCUSDT}"
DATA_DIR="${DATA_DIR:-data/offline_cache}"
export MODELS_DIR=""
# Default intervals (env override edilebilir)
INTERVALS="${INTERVALS:-1m,3m,5m,15m,30m,1h}"

# Label/SGD defaults (env override)
LABEL_HORIZON="${LABEL_HORIZON:-3}"
LABEL_THR="${LABEL_THR:-0.0005}"

SGD_ALPHA="${SGD_ALPHA:-1e-4}"
SGD_MAX_ITER="${SGD_MAX_ITER:-50}"
SGD_TOL="${SGD_TOL:-1e-3}"

# LSTM defaults
LSTM_INTERVALS="${LSTM_INTERVALS:-1m,3m,5m,15m,30m,1h}"
LSTM_SEQ_LEN_DEFAULT="${LSTM_SEQ_LEN_DEFAULT:-50}"
LSTM_EPOCHS="${LSTM_EPOCHS:-3}"
LSTM_BATCH_SIZE="${LSTM_BATCH_SIZE:-256}"

echo "[PIPELINE] SYMBOL=$SYMBOL"
echo "[PIPELINE] INTERVALS=$INTERVALS"
echo "[PIPELINE] LABEL_HORIZON=$LABEL_HORIZON LABEL_THR=$LABEL_THR"
echo "[PIPELINE] SGD_ALPHA=$SGD_ALPHA MAX_ITER=$SGD_MAX_ITER TOL=$SGD_TOL"
echo "[PIPELINE] LSTM_INTERVALS=$LSTM_INTERVALS SEQ_LEN_DEFAULT=$LSTM_SEQ_LEN_DEFAULT EPOCHS=$LSTM_EPOCHS BATCH=$LSTM_BATCH_SIZE"

echo
echo "==== 1) RESAMPLE 3m/30m ===="
SYMBOL="$SYMBOL" DATA_DIR="$DATA_DIR" python training/resample_offline_klines.py

echo
echo "==== 2) OFFLINE SGD TRAIN (ALL INTERVALS) ===="
SYMBOL="$SYMBOL" DATA_DIR="$DATA_DIR" MODELS_DIR="$MODELS_DIR" \
INTERVALS="$INTERVALS" \
LABEL_HORIZON="$LABEL_HORIZON" LABEL_THR="$LABEL_THR" \
SGD_ALPHA="$SGD_ALPHA" SGD_MAX_ITER="$SGD_MAX_ITER" SGD_TOL="$SGD_TOL" \
LSTM_INTERVALS="$LSTM_INTERVALS" LSTM_SEQ_LEN_DEFAULT="$LSTM_SEQ_LEN_DEFAULT" \
python training/offline_train_hybrid.py

echo
echo "==== 3) LSTM TRAIN (only intervals in LSTM_INTERVALS) ===="
IFS=',' read -ra ITVS <<< "$LSTM_INTERVALS"
for itv in "${ITVS[@]}"; do
  itv="$(echo "$itv" | xargs)"
  [ -z "$itv" ] && continue
  echo "[LSTM] interval=$itv"
  SYMBOL="$SYMBOL" DATA_DIR="$DATA_DIR" MODELS_DIR="$MODELS_DIR" \
  INTERVAL="$itv" \
  LABEL_HORIZON="$LABEL_HORIZON" LABEL_THR="$LABEL_THR" \
  LSTM_SEQ_LEN_DEFAULT="$LSTM_SEQ_LEN_DEFAULT" LSTM_EPOCHS="$LSTM_EPOCHS" LSTM_BATCH_SIZE="$LSTM_BATCH_SIZE" \
  python training/train_lstm_for_interval_simple.py
done

echo
echo "==== 4) VERIFY ARTIFACTS ===="
python - <<'PY'
import os, json
SYMBOL=os.getenv("SYMBOL","BTCUSDT")
from app_paths import MODELS_DIR
INTERVALS=os.getenv("INTERVALS","1m,3m,5m,15m,30m,1h").split(",")

def p(*a): print(*a)

ok=True
for itv in [x.strip() for x in INTERVALS if x.strip()]:
    m = os.path.join(MODELS_DIR, f"online_model_{itv}_best.joblib")
    meta = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
    p(f"\n[{itv}]")
    p("  model:", "OK" if os.path.exists(m) else "MISSING", m)
    p("  meta :", "OK" if os.path.exists(meta) else "MISSING", meta)
    if os.path.exists(meta):
        d=json.load(open(meta,"r",encoding="utf-8"))
        p(f"  best_auc={d.get('best_auc')} use_lstm_hybrid={d.get('use_lstm_hybrid')} seq_len={d.get('seq_len')}")
        p(f"  lstm_long_auc={d.get('lstm_long_auc')} lstm_short_auc={d.get('lstm_short_auc')}")
    # lstm artifacts
    longp=os.path.join(MODELS_DIR,f"lstm_long_{itv}.h5")
    shortp=os.path.join(MODELS_DIR,f"lstm_short_{itv}.h5")
    scal=os.path.join(MODELS_DIR,f"lstm_scaler_{itv}.joblib")
    p("  lstm_long :", "OK" if os.path.exists(longp) else "MISSING")
    p("  lstm_short:", "OK" if os.path.exists(shortp) else "MISSING")
    p("  scaler    :", "OK" if os.path.exists(scal) else "MISSING")
PY

echo
echo "[PIPELINE] DONE."
