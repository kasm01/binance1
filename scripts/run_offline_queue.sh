#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH="$PWD"
set -a; source .env; set +a
mkdir -p logs/training

# Eğitim sırası: her coin için 3m/5m/15m/30m/1h
intervals=(3m 5m 15m 30m 1h)

echo "[QUEUE] START  $(date -u +%F_%T)Z" | tee -a logs/training/queue_master.log
echo "[QUEUE] SYMBOLS=$SYMBOLS" | tee -a logs/training/queue_master.log
echo "[QUEUE] intervals=${intervals[*]}" | tee -a logs/training/queue_master.log

IFS=',' read -r -a syms <<< "${SYMBOLS:-BTCUSDT}"

for sym in "${syms[@]}"; do
  sym="$(echo "$sym" | xargs)"
  for itv in "${intervals[@]}"; do
    ts="$(date +%Y%m%d_%H%M%S)"
    LOG="logs/training/offline_train_${sym}_${itv}_${ts}.log"

    echo "===================================================" | tee -a "$LOG"
    echo "[RUN] SYMBOL=$sym INTERVAL=$itv HYBRID_MODE=1" | tee -a "$LOG"
    echo "[RUN] $(date -u +%FT%T)Z" | tee -a "$LOG"
    echo "===================================================" | tee -a "$LOG"

    # NOT: offline_train_hybrid.py scriptinin argümanları sizde nasıl ise ona göre ayarlanır.
    # Eğer argüman kabul etmiyorsa, env ile besliyoruz.
    SYMBOL="$sym" INTERVAL="$itv" HYBRID_MODE=1 \
      python -u training/offline_train_hybrid.py 2>&1 | tee -a "$LOG"

    ec=${PIPESTATUS[0]}
    echo "[DONE] SYMBOL=$sym INTERVAL=$itv exit_code=$ec at $(date -u +%FT%T)Z" | tee -a "$LOG"
    echo "[QUEUE] done $sym $itv (ec=$ec) log=$LOG" | tee -a logs/training/queue_master.log
  done
done

echo "[QUEUE] FINISH $(date -u +%F_%T)Z" | tee -a logs/training/queue_master.log
