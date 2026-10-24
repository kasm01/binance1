#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ./scripts/orch_lib.sh

STATE_FILE="run/short_orch_health.state"
FAIL_THRESHOLD="${SHORT_ORCH_HEALTH_FAIL_THRESHOLD:-2}"
RESTART_COOLDOWN_SEC="${SHORT_ORCH_HEALTH_RESTART_COOLDOWN_SEC:-120}"

if ! systemctl --user is-active --quiet binance1-orch.service; then
  exit 0
fi

FAIL_COUNT=0
LAST_RESTART=0
[[ -f "$STATE_FILE" ]] && source "$STATE_FILE" || true
FAIL_COUNT="${FAIL_COUNT:-0}"
LAST_RESTART="${LAST_RESTART:-0}"

save_state() {
  cat >"${STATE_FILE}.tmp" <<EOF
FAIL_COUNT=${FAIL_COUNT}
LAST_RESTART=${LAST_RESTART}
EOF
  mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
}

missing=()
for name in aggregator top_selector master_executor intent_bridge; do
  expect="$(expected_cmd "$name")"
  pid="$(find_project_pid "$name" "$expect")"
  if [[ -z "${pid:-}" ]]; then
    missing+=("$name")
  else
    echo "$pid" > "run/${name}.pid"
  fi
done

if [[ ${#missing[@]} -eq 0 ]]; then
  FAIL_COUNT=0
  save_state
  echo "[SHORT-ORCH-HEALTH][OK] all workers alive"
  exit 0
fi

FAIL_COUNT=$((FAIL_COUNT + 1))
now="$(date +%s)"
echo "[SHORT-ORCH-HEALTH][WARN] missing=${missing[*]} fail=${FAIL_COUNT}/${FAIL_THRESHOLD}"

if (( FAIL_COUNT < FAIL_THRESHOLD )); then
  save_state
  exit 0
fi

if (( LAST_RESTART > 0 && now - LAST_RESTART < RESTART_COOLDOWN_SEC )); then
  save_state
  echo "[SHORT-ORCH-HEALTH][WARN] restart cooldown active"
  exit 0
fi

FAIL_COUNT=0
LAST_RESTART="$now"
save_state
logger -t binance1-short-orch-health "restarting missing workers: ${missing[*]}" || true
systemctl --user restart binance1-orch.service
