#!/usr/bin/env bash
set -e

cd ~/binance1
source venv/bin/activate

echo "[HYBRID-LOOP] Başladı. 24/7 offline LSTM+SGD eğitim döngüsü."

while true; do
  for INTERVAL in 1m 5m 15m 1h; do
    TS="$(date '+%Y-%m-%d %H:%M:%S')"
    LOG_FILE="logs/offline_${INTERVAL}_$(date +%F).log"

    echo "[${TS}][HYBRID-LOOP] ${INTERVAL} için offline_pretrain_six_months (deep + hybrid) başlıyor..." | tee -a "$LOG_FILE"

    HYBRID_MODE=true INTERVAL="${INTERVAL}" ./venv/bin/python -m training.offline_pretrain_six_months --mode deep >> "$LOG_FILE" 2>&1

    TS_END="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${TS_END}][HYBRID-LOOP] ${INTERVAL} offline eğitim bitti." | tee -a "$LOG_FILE"
  done

  # Tüm interval’lar bittikten sonra bekleme süresi (ör: 12 saat)
  TS_SLEEP="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${TS_SLEEP}][HYBRID-LOOP] Tüm interval eğitimleri bitti. 12 saat uyuyor..." | tee -a "logs/offline_loop.log"
  sleep 43200  # 12 saat
done
