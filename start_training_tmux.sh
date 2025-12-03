#!/usr/bin/env bash

SESSION_NAME="binance_training"
PYTHON="/home/kasm920/binance1/venv/bin/python"

cd "$(dirname "$0")"

# 1) Yeni session
tmux new-session -d -s "$SESSION_NAME"

# 2) 1m eğitim
tmux send-keys -t "$SESSION_NAME":0.0 "INTERVAL=1m TRAINING_MODE=true $PYTHON main.py" C-m

# 3) 5m eğitim
tmux split-window -v -t "$SESSION_NAME":0.0
tmux send-keys -t "$SESSION_NAME":0.1 "INTERVAL=5m TRAINING_MODE=true $PYTHON main.py" C-m

# 4) 15m eğitim
tmux split-window -v -t "$SESSION_NAME":0.1
tmux send-keys -t "$SESSION_NAME":0.2 "INTERVAL=15m TRAINING_MODE=true $PYTHON main.py" C-m

# 5) 1h eğitim
tmux split-window -v -t "$SESSION_NAME":0.2
tmux send-keys -t "$SESSION_NAME":0.3 "INTERVAL=1h TRAINING_MODE=true $PYTHON main.py" C-m

# 6) hybrid eğitim
tmux split-window -v -t "$SESSION_NAME":0.3
tmux send-keys -t "$SESSION_NAME":0.4 "INTERVAL=hybrid TRAINING_MODE=true $PYTHON main.py" C-m

# 7) Layout'u tiled yap
tmux select-layout -t "$SESSION_NAME" tiled

# 8) Session'a bağlan
tmux attach-session -t "$SESSION_NAME"
