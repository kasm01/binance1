#!/usr/bin/env bash
cd ~/binance1

# Eğitim loop'u
if ! tmux has-session -t hybrid-train 2>/dev/null; then
  tmux new-session -d -s hybrid-train 'cd ~/binance1 && ./training/run_hybrid_training_loop.sh'
  echo "[START-ALL] hybrid-train tmux session başlatıldı."
else
  echo "[START-ALL] hybrid-train zaten çalışıyor."
fi

# 1m canlı bot
if ! tmux has-session -t bot-1m 2>/dev/null; then
  tmux new-session -d -s bot-1m 'cd ~/binance1 && INTERVAL=1m TRAINING_MODE=true HYBRID_MODE=true ./venv/bin/python main.py'
  echo "[START-ALL] bot-1m tmux session başlatıldı."
else
  echo "[START-ALL] bot-1m zaten çalışıyor."
fi
