#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# DISARM: force dry-run and enable kill switch
perl -0777 -i -pe '
  s/^ARMED=.*/ARMED=0/m or $_ .= "\nARMED=0\n";
  s/^LIVE_KILL_SWITCH=.*/LIVE_KILL_SWITCH=1/m or $_ .= "LIVE_KILL_SWITCH=1\n";
' .env

echo "[DISARM] ARMED=0 LIVE_KILL_SWITCH=1 (saved to .env)"
