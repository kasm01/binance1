#!/usr/bin/env bash
cd ~/binance1
source venv/bin/activate

INTERVAL=15m HYBRID_MODE=true ./venv/bin/python -m training.offline_pretrain_six_months \
  --mode light \
  --months 1 \
  | tee logs/offline_15m_$(date +%F).log
