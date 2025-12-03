#!/usr/bin/env bash

SESSION_NAME="binance_training"
PROJECT_DIR="$HOME/binance1"
PYTHON="$PROJECT_DIR/venv/bin/python"

cd "$PROJECT_DIR" || { echo "Proje klasörü bulunamadı: $PROJECT_DIR"; exit 1; }

# Session zaten varsa uyarıp çık
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Tmux session zaten var: $SESSION_NAME"
  echo "Bağlanmak için: tmux attach -t $SESSION_NAME"
  exit 0
fi

# ───────── 1) Yeni session: 1m ─────────
tmux new-session -d -s "$SESSION_NAME" -n "1m" \
  "cd $PROJECT_DIR && INTERVAL=1m TRAINING_MODE=true $PYTHON main.py"

# ───────── 2) Aynı pencerede split ile 5m ─────────
tmux split-window -v -t "$SESSION_NAME:0" \
  "cd $PROJECT_DIR && INTERVAL=5m TRAINING_MODE=true $PYTHON main.py"

# ───────── 3) 15m için yeni yatay split ─────────
tmux split-window -h -t "$SESSION_NAME:0.0" \
  "cd $PROJECT_DIR && INTERVAL=15m TRAINING_MODE=true $PYTHON main.py"

# ───────── 4) 1h için diğer pane ─────────
tmux split-window -h -t "$SESSION_NAME:0.1" \
  "cd $PROJECT_DIR && INTERVAL=1h TRAINING_MODE=true $PYTHON main.py"

# Layout'u düzenle (tiled)
tmux select-layout -t "$SESSION_NAME:0" tiled

# ───────── 5) Hybrid için ayrı pencere ─────────
tmux new-window -t "$SESSION_NAME:1" -n "hybrid" \
  "cd $PROJECT_DIR && INTERVAL=1m TRAINING_MODE=true HYBRID_MODE=true $PYTHON main.py"

echo "Tmux eğitim session'ı başlatıldı: $SESSION_NAME"
echo "Bağlanmak için: tmux attach -t $SESSION_NAME"
