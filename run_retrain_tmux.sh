#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
SYMBOLS=("BTCUSDT" "ETHUSDT" "BNBUSDT" "SOLUSDT")
INTERVALS=("1m" "3m" "5m" "15m" "1h")

OFFLINE_DIR="${OFFLINE_DIR:-data/offline_cache}"
MODELS_DIR="${MODELS_DIR:-models}"
LOG_DIR="${LOG_DIR:-logs/training}"

# eğitimde daha büyük limit iyi olur (csv 6m ise zaten sınırlı)
DATA_LIMIT="${BT_DATA_LIMIT:-60000}"
MODEL_WINDOW="${BT_MODEL_WINDOW:-200}"

# LSTM eğitimi: 1 yaparsan çalışır
TRAIN_LSTM="${TRAIN_LSTM:-0}"

# Preprocess flags -> train & inference aynı kalsın diye
export BT_TIME_NORM="${BT_TIME_NORM:-0}"
export BT_LOG1P_FEATURES="${BT_LOG1P_FEATURES:-0}"
export BT_PRICE_LOG="${BT_PRICE_LOG:-1}"

# CPU stabilite
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

mkdir -p "$LOG_DIR" "$MODELS_DIR"

has_py() { [[ -f "$1" ]]; }

SGD_HELPER_SCRIPT="training/train_sgd_helper.py"
RESAMPLE_SCRIPT="training/resample_offline_klines.py"
LSTM_ONE_SCRIPT="training/train_lstm_for_interval.py"
LSTM_ALL_SCRIPT="training/train_lstm_all.py"

if ! has_py "$SGD_HELPER_SCRIPT"; then
  echo "[ERROR] missing: $SGD_HELPER_SCRIPT"
  exit 1
fi

echo "[RETRAIN] symbols: ${SYMBOLS[*]}"
echo "[RETRAIN] intervals: ${INTERVALS[*]}"
echo "[RETRAIN] TRAIN_LSTM=$TRAIN_LSTM"
echo "[RETRAIN] preprocess: BT_TIME_NORM=$BT_TIME_NORM BT_LOG1P_FEATURES=$BT_LOG1P_FEATURES BT_PRICE_LOG=$BT_PRICE_LOG"

# =========================
# helper: ensure CSV exists
# =========================
ensure_csv () {
  local sym="$1"
  local itv="$2"
  local csv="$OFFLINE_DIR/${sym}_${itv}_6m.csv"

  if [[ -f "$csv" ]]; then
    echo "$csv"
    return 0
  fi

  # yoksa resample ile üretmeyi dene (1m'den üretim)
  if [[ "$itv" != "1m" && -f "$OFFLINE_DIR/${sym}_1m_6m.csv" && -f "$RESAMPLE_SCRIPT" ]]; then
    echo "[INFO] missing $csv -> resample from ${sym}_1m_6m.csv using $RESAMPLE_SCRIPT"
    PYTHONPATH="$PWD" python -u "$RESAMPLE_SCRIPT" \
      --in_csv "$OFFLINE_DIR/${sym}_1m_6m.csv" \
      --out_csv "$csv" \
      --interval "$itv" || true
  fi

  if [[ -f "$csv" ]]; then
    echo "$csv"
    return 0
  fi

  echo ""
  return 1
}

# =========================
# MAIN LOOP
# =========================
for itv in "${INTERVALS[@]}"; do
  for sym in "${SYMBOLS[@]}"; do
    echo "============================================================"
    echo "[RETRAIN] symbol=$sym interval=$itv"
    echo "============================================================"

    CSV="$(ensure_csv "$sym" "$itv" || true)"
    if [[ -z "${CSV:-}" ]]; then
      echo "[WARN] offline csv yok: $OFFLINE_DIR/${sym}_${itv}_6m.csv (resample da üretemedi)"
      continue
    fi

    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="$LOG_DIR/retrain_${sym}_${itv}_${TS}.log"

    export BT_SYMBOL="$sym"
    export BT_MAIN_INTERVAL="$itv"
    export SGD_TRAIN_CSV="$CSV"
    export BT_DATA_LIMIT="$DATA_LIMIT"
    export BT_MODEL_WINDOW="$MODEL_WINDOW"

    # SGD train hyperparams (mevcut değerler sende iyi duruyor)
    export SGD_ALPHA="${SGD_ALPHA:-5e-5}"
    export SGD_MAX_ITER="${SGD_MAX_ITER:-20000}"
    export SGD_TOL="${SGD_TOL:-1e-5}"
    export SGD_LABEL_H="${SGD_LABEL_H:-3}"
    export SGD_LABEL_THR="${SGD_LABEL_THR:-0.0005}"

    echo "[RUN] $SGD_HELPER_SCRIPT -> $LOG"
    PYTHONPATH="$PWD" python -u "$SGD_HELPER_SCRIPT" 2>&1 | tee "$LOG"
    echo "[DONE] SGD $sym $itv"

    if [[ "$TRAIN_LSTM" == "1" ]]; then
      TS2="$(date +%Y%m%d_%H%M%S)"
      LOG2="$LOG_DIR/lstm_${sym}_${itv}_${TS2}.log"

      if [[ -f "$LSTM_ONE_SCRIPT" ]]; then
        echo "[RUN] $LSTM_ONE_SCRIPT (symbol=$sym interval=$itv) -> $LOG2"
        PYTHONPATH="$PWD" python -u "$LSTM_ONE_SCRIPT" --symbol "$sym" --interval "$itv" 2>&1 | tee "$LOG2"
        echo "[DONE] LSTM $sym $itv"
      elif [[ -f "$LSTM_ALL_SCRIPT" ]]; then
        echo "[RUN] $LSTM_ALL_SCRIPT -> $LOG2"
        PYTHONPATH="$PWD" python -u "$LSTM_ALL_SCRIPT" 2>&1 | tee "$LOG2"
        echo "[DONE] LSTM ALL"
      else
        echo "[WARN] LSTM script bulunamadı (train_lstm_for_interval.py / train_lstm_all.py yok). Atlıyorum."
      fi
    fi
  done
done

echo "============================================================"
echo "[ALL DONE] retrain finished."
echo "Models: $MODELS_DIR"
echo "Logs:   $LOG_DIR"
echo "============================================================"
