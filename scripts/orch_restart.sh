#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Usage:
#   ./scripts/orch_restart.sh
# Options:
#   QUIET_STOP=1         -> stop çıktısını sustur
#   WAIT_SEC=2           -> start sonrası bekleme (default 2)
#   STATUS_HEAD=120      -> status çıktısının ilk N satırı (default 120)
#   CHECK_READY=1        -> orch_start.sh içindeki readiness check'e güven (default 1)

WAIT_SEC="${WAIT_SEC:-2}"
STATUS_HEAD="${STATUS_HEAD:-120}"
CHECK_READY="${CHECK_READY:-1}"

# Stop (quiet optional)
if [[ "${QUIET_STOP:-0}" == "1" ]]; then
  QUIET_STOP=1 ./scripts/orch_stop.sh >/dev/null 2>&1 || true
else
  ./scripts/orch_stop.sh || true
fi

# Start
./scripts/orch_start.sh

# Wait
sleep "$WAIT_SEC"

# Status
./scripts/orch_status.sh | head -n "$STATUS_HEAD"
