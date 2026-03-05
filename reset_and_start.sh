#!/bin/bash

echo "==============================="
echo " BINANCE1 BOT FULL RESET"
echo "==============================="

cd ~/binance1 || exit

echo ""
echo "1️⃣ Killing running processes..."

pkill -f "python.*main.py" || true
pkill -f "intent_bridge.py" || true
pkill -f "scanner" || true
pkill -f "executor" || true
pkill -f "top_selector" || true
pkill -f "master_executor" || true
pkill -f "binance1" || true

sleep 1

echo ""
echo "2️⃣ Stopping systemd services..."

systemctl --user stop binance1.service 2>/dev/null || true
systemctl --user stop binance1-orch.service 2>/dev/null || true
systemctl --user stop binance1-main.service 2>/dev/null || true

echo ""
echo "3️⃣ Cleaning Redis streams/state..."

redis-cli -n 0 DEL trade_intents_stream
redis-cli -n 0 DEL exec_events_stream
redis-cli -n 0 DEL open_positions_state
redis-cli -n 0 DEL open_positions_state:close_cd
redis-cli -n 0 DEL positions

echo ""
echo "4️⃣ Cleaning python cache..."

find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

echo ""
echo "5️⃣ Activating venv..."

source venv/bin/activate

echo ""
echo "6️⃣ Loading environment..."

set -a
source .env
set +a

export DRY_RUN=1
export ARMED=0
export LIVE_KILL_SWITCH=1
export BINANCE_TESTNET=0

echo ""
echo "7️⃣ Checking python files..."

python -m py_compile orchestration/executor/intent_bridge.py
python -m py_compile core/trade_executor.py
python -m py_compile main.py

echo ""
echo "8️⃣ Starting IntentBridge..."

mkdir -p logs/orch

nohup ./venv/bin/python -u orchestration/executor/intent_bridge.py \
> logs/orch/intent_bridge.log 2>&1 &

sleep 2

echo ""
echo "9️⃣ Starting MAIN BOT..."

./venv/bin/python -u main.py &
sleep 3

echo ""
echo "🔟 Sending test intent..."

redis-cli -n 0 XADD trade_intents_stream "*" json \
'{"items":[{"symbol":"BTCUSDT","side":"long","interval":"1m","intent_id":"reset-test-open","price":65000,"trail_pct":0.03,"stall_ttl_sec":600}]}'

sleep 1

redis-cli -n 0 XADD trade_intents_stream "*" json \
'{"items":[{"symbol":"BTCUSDT","side":"close","interval":"1m","intent_id":"reset-test-close"}]}'

echo ""
echo "==============================="
echo " BOT STARTED"
echo "==============================="

echo ""
echo "Check logs:"
echo "tail -f logs/orch/intent_bridge.log"
