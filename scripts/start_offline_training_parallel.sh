#!/bin/bash
set -e

PROJECT_ROOT="$HOME/binance1"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

INTERVALS=("1m" "5m" "15m" "1h")
MODE="${1:-deep}"     # default: deep
USE_LSTM_HYBRID="--use-lstm-hybrid"  # istersen boş bırakabilirsin

echo "==========================================="
echo " OFFLINE PARALLEL TRAINING BAŞLIYOR"
echo " MODE           : $MODE"
echo " INTERVALS      : ${INTERVALS[*]}"
echo " PROJE DİZİNİ   : $PROJECT_ROOT"
echo " LOG DİZİNİ     : $LOG_DIR"
echo "==========================================="

cd "$PROJECT_ROOT"

START_TS=$(date +%s)

pids=()

for interval in "${INTERVALS[@]}"; do
  LOG_FILE="$LOG_DIR/offline_${interval}.log"

  echo ""
  echo ">>> [LAUNCH] interval=${interval} | log=${LOG_FILE}"

  (
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OFFLINE TRAINING STARTED | interval=${interval} | mode=${MODE}" >> "$LOG_FILE"

    interval_start_ts=$(date +%s)

    # Burada tek interval çalıştırıyoruz;
    # eğitim script'i zaten long/short + all rounds yapıyor.
    python -m training.offline_pretrain_six_months \
      --mode "$MODE" \
      --intervals "$interval" \
      $USE_LSTM_HYBRID >> "$LOG_FILE" 2>&1

    interval_end_ts=$(date +%s)
    interval_elapsed=$((interval_end_ts - interval_start_ts))

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OFFLINE TRAINING FINISHED | interval=${interval} | elapsed=${interval_elapsed}s" >> "$LOG_FILE"
  ) &

  pids+=($!)
done

echo ""
echo "Tüm interval eğitimleri arka planda başlatıldı. PIDs: ${pids[*]}"
echo "Loglar: $LOG_DIR/offline_<interval>.log"

# Tüm jobların bitmesini bekle
for pid in "${pids[@]}"; do
  wait "$pid"
done

END_TS=$(date +%s)
TOTAL_ELAPSED=$((END_TS - START_TS))

echo ""
echo "==========================================="
echo " TÜM OFFLINE EĞİTİMLER BİTTİ"
echo " TOPLAM SÜRE: ${TOTAL_ELAPSED} saniye (~$((TOTAL_ELAPSED / 60)) dk)"
echo " Loglar: $LOG_DIR/offline_*.log"
echo "==========================================="

# Telegram bildirimi (isteğe bağlı)
if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
  MSG="Offline training tamamlandı.%0AMode: ${MODE}%0AIntervals: ${INTERVALS[*]}%0AToplam süre: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60)) dk)"
  curl -s \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}&text=${MSG}" >/dev/null 2>&1 || true
  echo "Telegram bildirimi gönderildi."
else
  echo "Telegram bildirimi gönderilmedi (TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID set değil)."
fi
