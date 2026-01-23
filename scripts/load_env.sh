#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ENV] file not found: $ENV_FILE"
  exit 1
fi

# 1) .env içindeki "KEY: value" satırlarını "KEY=value" formatına çevir (geçici dosyaya)
# 2) Yorum/boş satırları atla
# 3) Export edilebilir güvenli satırları bırak
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# normalize: "KEY: value" -> "KEY=value"
# ayrıca "export KEY=value" desteklenir
sed -E \
  -e 's/^[[:space:]]+//; s/[[:space:]]+$//' \
  -e '/^$/d' \
  -e '/^[#;]/d' \
  -e 's/^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*:[[:space:]]*/\1=/' \
  "$ENV_FILE" > "$tmp"

set -a
# shellcheck disable=SC1090
source "$tmp"
set +a

echo "[ENV] loaded from $ENV_FILE (normalized)"
