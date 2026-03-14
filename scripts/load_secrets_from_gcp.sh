#!/usr/bin/env bash
set -euo pipefail

OUT="/home/kasm920/binance1/.runtime_secrets.env"
TMP="${OUT}.tmp"

get_secret() {
  local name="$1"
  gcloud secrets versions access latest --secret="$name" 2>/dev/null | tr -d '\r' | paste -sd '' -
}

write_secret() {
  local name="$1"
  local value
  value="$(get_secret "$name")"
  printf '%s=%s\n' "$name" "$value" >> "$TMP"
}

: > "$TMP"

write_secret ALCHEMY_ETH_API_KEY
write_secret ARBI_API_KEY
write_secret BINANCE_API_KEY
write_secret BINANCE_API_SECRET
write_secret BINANCE_TESTNET_API_KEY
write_secret BINANCE_TESTNET_API_SECRET
write_secret BSCSCAN_API_KEY
write_secret COINGLASS_API_KEY
write_secret COINMARKETCAP_API_KEY
write_secret CRYPTOQUANT_API_KEY
write_secret ETH_API_KEY
write_secret ETHERSCAN_API_KEY
write_secret GRAPH_API_KEY
write_secret INFURA_API_KEY
write_secret OKX_API_KEY
write_secret OKX_API_SECRET
write_secret OKX_PASSPHRASE
write_secret PG_DSN
write_secret POLYGON_API_KEY
write_secret SANTIMENT_API_KEY
write_secret TELEGRAM_ALLOWED_CHAT_IDS
write_secret TELEGRAM_BOT_TOKEN
write_secret TELEGRAM_CHAT_ID
write_secret THE_GRAPH_API_KEY

chmod 600 "$TMP"
mv "$TMP" "$OUT"

echo "Secrets loaded into $OUT"
