#!/usr/bin/env bash
set -euo pipefail

BASE="/home/kasm920/binance1"
STATE_FILE="$BASE/logs/logrotate.status"
CONF_FILE="$BASE/scripts/logrotate_binance1.conf"

/usr/sbin/logrotate -s "$STATE_FILE" "$CONF_FILE"
find "$BASE/logs" -type f -name "*.gz" -mtime +10 -delete
