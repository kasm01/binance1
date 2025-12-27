#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source venv/bin/activate
mkdir -p logs/training

export SYMBOL="${SYMBOL:-BTCUSDT}"
export LSTM_HORIZON="${LSTM_HORIZON:-1}"
export LSTM_WINDOW="${LSTM_WINDOW:-50}"

intervals=(1m 5m 15m 1h)

for itv in "${intervals[@]}"; do
  export INTERVAL="$itv"
  LOG="logs/training/lstm_${SYMBOL}_${INTERVAL}_h${LSTM_HORIZON}_w${LSTM_WINDOW}_$(date +%Y%m%d_%H%M%S).log"
  echo "===== START interval=${INTERVAL} symbol=${SYMBOL} horizon=${LSTM_HORIZON} window=${LSTM_WINDOW} | log=${LOG} ====="

  PYTHONPATH="$PWD" python -u training/train_lstm_for_interval.py \
    2>&1 | tee "$LOG"

  echo "===== DONE interval=${INTERVAL} ====="
done

echo "ALL DONE."
