#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/orch_stop.sh || true
sleep 0.5
./scripts/orch_start.sh
