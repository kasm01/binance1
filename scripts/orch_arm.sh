#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ARM: enable live ability (still requires DRY_RUN=0)
tok="$(python - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"

# .env içinde ARMED=1, ARM_TOKEN=..., LIVE_KILL_SWITCH=0 yap
perl -0777 -i -pe '
  s/^ARMED=.*/ARMED=1/m or $_ .= "\nARMED=1\n";
  s/^ARM_TOKEN=.*/ARM_TOKEN='"$tok"'/m or $_ .= "ARM_TOKEN='"$tok"'\n";
  s/^LIVE_KILL_SWITCH=.*/LIVE_KILL_SWITCH=0/m or $_ .= "LIVE_KILL_SWITCH=0\n";
' .env

echo "[ARM] ARMED=1 ARM_TOKEN=$tok (saved to .env)"
echo "[ARM] Note: you must also set DRY_RUN=0 for real execution."
