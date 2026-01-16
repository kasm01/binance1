#!/usr/bin/env bash

SESSION_NAME="binance_training"
PROJECT_DIR="$HOME/binance1"
PYTHON="$PROJECT_DIR/venv/bin/python"

cd "$PROJECT_DIR" || { echo "Proje klasörü bulunamadı: $PROJECT_DIR"; exit 1; }

# Session zaten varsa uyar
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Tmux session zaten var: $SESSION_NAME"
  echo "Bağlanmak için: tmux attach -t $SESSION_NAME"
  exit 0
fi

mkdir -p logs/training logs/data

# Ortak env flag'ler
COMMON_ENV="TRAINING_MODE=true HYBRID_MODE=true USE_MTF_ENS=false DRY_RUN=true"

# ==========================================================
# 1️⃣ CANLI TRAINING WINDOW (1m / 5m / 15m / 1h)
# ==========================================================
tmux new-session -d -s "$SESSION_NAME" -n "live_train" \
  "cd $PROJECT_DIR && INTERVAL=1m $COMMON_ENV $PYTHON main.py"

tmux split-window -v -t "$SESSION_NAME:0" \
  "cd $PROJECT_DIR && INTERVAL=5m $COMMON_ENV $PYTHON main.py"

tmux split-window -h -t "$SESSION_NAME:0.0" \
  "cd $PROJECT_DIR && INTERVAL=15m $COMMON_ENV $PYTHON main.py"

tmux split-window -h -t "$SESSION_NAME:0.1" \
  "cd $PROJECT_DIR && INTERVAL=1h $COMMON_ENV $PYTHON main.py"

tmux select-layout -t "$SESSION_NAME:0" tiled

# ==========================================================
# 2️⃣ HYBRID + MTF ENSEMBLE WINDOW
# ==========================================================
tmux new-window -t "$SESSION_NAME:1" -n "hybrid_mtf" \
  "cd $PROJECT_DIR && \
   INTERVAL=1m TRAINING_MODE=true HYBRID_MODE=true USE_MTF_ENS=true DRY_RUN=true \
   $PYTHON main.py"

# ==========================================================
# 3️⃣ WHALE TRAINING WINDOW
# ==========================================================
tmux new-window -t "$SESSION_NAME:2" -n "whale_train" \
  "cd $PROJECT_DIR && \
   for TF in 1m 5m 15m 1h; do \
     echo \"[WHALE-TRAIN] Interval=\$TF\"; \
     SYMBOL=BTCUSDT INTERVAL=\$TF $PYTHON training/train_whale_for_interval.py; \
     echo \"[WHALE-TRAIN] Interval=\$TF tamamlandı\"; \
   done; \
   echo '[WHALE] Tüm interval eğitimleri tamamlandı'; read"

# ==========================================================
# 4️⃣ SGD STRICT OFFLINE TRAINING (6 AY)
# ==========================================================
tmux new-window -t "$SESSION_NAME:3" -n "sgd_6m_strict" \
  "cd $PROJECT_DIR && \
   export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 && \
   export SGD_TRAIN_CSV=data/offline_cache/BTCUSDT_5m_6m.csv && \
   export SGD_OUT_DIR=models && \
   export SGD_INTERVAL=5m && \
   export SGD_LABEL_H=3 && \
   export SGD_LABEL_THR=0.0005 && \
   export SGD_MAX_ITER=120000 && \
   export SGD_ALPHA=1e-5 && \
   export SGD_TOL=1e-6 && \
   LOG=logs/training/sgd_6m_strict_\$(date +%Y%m%d_%H%M%S).log && \
   echo \"[SGD] 6 aylık strict eğitim başlıyor...\" && \
   PYTHONPATH=$PROJECT_DIR $PYTHON -u training/train_sgd_online_strict.py 2>&1 | tee \$LOG && \
   # --- AUC-POST-SGD: retrain sonrası (1 kez) hourly auc_used append ---
   $PYTHON - <<'PY'
import os
from utils.auc_history import seed_auc_history_if_missing, append_auc_used_once_per_hour
mtf = os.getenv('MTF_INTERVALS','1m,3m,5m,15m,30m,1h')
intervals = [x.strip() for x in mtf.split(',') if x.strip()]
seed_auc_history_if_missing(intervals=intervals, logger=None)
append_auc_used_once_per_hour(intervals=intervals, logger=None)
print('[AUC-POST-SGD] hourly auc_used appended for:', intervals)
PY
   # --- /AUC-POST-SGD ---
   && \
   echo \"[SGD] Eğitim tamamlandı. Log: \$LOG\" && read"



# ==========================================================
# BİTTİ
# ==========================================================
echo "✅ Tmux eğitim session'ı başlatıldı: $SESSION_NAME"
echo "➡️  Bağlanmak için: tmux attach -t $SESSION_NAME"
