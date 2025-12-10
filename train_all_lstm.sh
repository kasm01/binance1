#!/bin/bash

cd ~/binance1
source venv/bin/activate

mkdir -p logs

echo "=== 1m eğitimi başlıyor ==="
python models/train_lstm_for_interval.py --interval 1m | tee logs/train_lstm_1m.log

echo "=== 5m eğitimi başlıyor ==="
python models/train_lstm_for_interval.py --interval 5m | tee logs/train_lstm_5m.log

echo "=== 15m eğitimi başlıyor ==="
python models/train_lstm_for_interval.py --interval 15m | tee logs/train_lstm_15m.log

echo "=== 1h eğitimi başlıyor ==="
python models/train_lstm_for_interval.py --interval 1h | tee logs/train_lstm_1h.log

echo "=== TÜM EĞİTİMLER TAMAMLANDI ==="
