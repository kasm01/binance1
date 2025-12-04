#!/usr/bin/env bash
set -e

cd ~/binance1
source venv/bin/activate

INTERVAL=1h HYBRID_MODE=true ./venv/bin/python -m training.offline_pretrain_six_months \
  --mode light \
  --months 1 \
  | tee logs/offline_1h_$(date +%F).log
