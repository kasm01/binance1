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

# Ortak env flag'ler (eğitim için güvenli)
COMMON_ENV="TRAINING_MODE=true HYBRID_MODE=true USE_MTF_ENS=false DRY_RUN=true"

# ───────── 1) Yeni session: 1m (eğitim) ─────────
tmux new-session -d -s "$SESSION_NAME" -n "train_1m" \
  "cd $PROJECT_DIR && INTERVAL=1m $COMMON_ENV $PYTHON main.py"

# ───────── 2) Aynı pencerede alt pane: 5m ─────────
tmux split-window -v -t "$SESSION_NAME:0" \
  "cd $PROJECT_DIR && INTERVAL=5m $COMMON_ENV $PYTHON main.py"

# ───────── 3) 15m için yeni yatay split ─────────
tmux split-window -h -t "$SESSION_NAME:0.0" \
  "cd $PROJECT_DIR && INTERVAL=15m $COMMON_ENV $PYTHON main.py"

# ───────── 4) 1h için diğer pane ─────────
tmux split-window -h -t "$SESSION_NAME:0.1" \
  "cd $PROJECT_DIR && INTERVAL=1h $COMMON_ENV $PYTHON main.py"

# Layout'u düzenle (tiled)
tmux select-layout -t "$SESSION_NAME:0" tiled

# ───────── 5) Hybrid/MTF için ayrı pencere ─────────
# Burada USE_MTF_ENS=true yapıyoruz; 1m'yi MTF ensemble + whale ile gözlemlemek için
tmux new-window -t "$SESSION_NAME:1" -n "hybrid_mtf" \
  "cd $PROJECT_DIR && INTERVAL=1m TRAINING_MODE=true HYBRID_MODE=true USE_MTF_ENS=true DRY_RUN=true $PYTHON main.py"

# ───────── 6) Whale training için ayrı pencere ─────────
tmux new-window -t "$SESSION_NAME:2" -n "whale" \
  "cd $PROJECT_DIR && for TF in 1m 5m 15m 1h; do \
     echo \"[WHALE-TRAIN] Interval=\$TF\"; \
     SYMBOL=BTCUSDT INTERVAL=\$TF $PYTHON training/train_whale_for_interval.py; \
     echo \"[WHALE-TRAIN] Interval=\$TF bitti\"; \
   done; echo 'Whale training tamamlandı. Pane içinde logları inceleyebilirsin.'; read"

echo "Tmux eğitim session'ı başlatıldı: $SESSION_NAME"
echo "Bağlanmak için: tmux attach -t $SESSION_NAME"
