#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== PID files ==="
ls -1 run/*.pid 2>/dev/null || echo "(none)"

echo
echo "=== Running orchestration processes (ps) ==="
# hem dosya yolu ile çalışanları hem -c Aggregator(...) çalışanı yakala
ps aux | egrep "orchestration/(scanners|selector|executor)|Aggregator\(RedisBus\)\.run_forever|orchestration\.aggregator\.aggregator" | grep -v grep || true

echo
echo "=== pgrep -af (orchestration / aggregator) ==="
pgrep -af "orchestration/|Aggregator\(RedisBus\)\.run_forever|orchestration\.aggregator\.aggregator" || true

echo
echo "=== Note ==="
echo "If you still don't see aggregator here, it's still OK if aggregator.log is alive and candidates_stream is growing."
