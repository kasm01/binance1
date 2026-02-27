#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/orch_lib.sh

LOGTAG="${LOGTAG:-[STOP]}"
QUIET_STOP="${QUIET_STOP:-0}"

msg() {
  if [[ "${QUIET_STOP}" == "1" ]]; then
    return 0
  fi
  echo "$@"
}

stop_from_pidfiles() {
  shopt -s nullglob

  # 1) Downstream önce (daha güvenli): intent_bridge -> master -> selector -> aggregator
  local name pidfile pid
  for name in intent_bridge master_executor top_selector aggregator; do
    pidfile="${RUNDIR}/${name}.pid"
    if [[ -f "${pidfile}" ]]; then
      pid="$(cat "${pidfile}" 2>/dev/null || true)"
      stop_pid "${name}" "${pid}"
      rm -f "${pidfile}" || true
    else
      msg "${LOGTAG} ${name} pidfile missing"
    fi
  done

  # 2) scanner_w* pidfile'ları
  for pidfile in "${RUNDIR}"/scanner_*.pid; do
    name="$(basename "${pidfile}" .pid)"
    pid="$(cat "${pidfile}" 2>/dev/null || true)"
    stop_pid "${name}" "${pid}"
    rm -f "${pidfile}" || true
  done

  shopt -u nullglob
}

stop_from_pgrep() {
  # pidfile kaçarsa diye ikinci güvenlik ağı
  # pkill -f yerine pid listesiyle stop_pid (daha kontrollü)

  local -a patterns=(
    "orchestration/executor/intent_bridge.py"
    "orchestration/executor/master_executor.py"
    "orchestration/selector/top_selector.py"
    "orchestration/aggregator/run_aggregator.py"
    "Aggregator\\(RedisBus\\(\\)\\)\\.run_forever"
    "orchestration/scanners/worker_stub.py"
    "python -m orchestration\\.executor\\.intent_bridge"
    "python -m orchestration\\.executor\\.master_executor"
    "python -m orchestration\\.selector\\.top_selector"
  )

  local pat
  for pat in "${patterns[@]}"; do
    # shellcheck disable=SC2207
    local pids=($(pgrep -af "${pat}" | awk '{print $1}' | sort -u || true))
    [[ "${#pids[@]}" -gt 0 ]] || continue

    local pid
    for pid in "${pids[@]}"; do
      stop_pid "pgrep:${pat}" "${pid}"
    done
  done
}

main() {
  stop_from_pidfiles
  stop_from_pgrep

  # Son temizlik: pidfile bırakma
  rm -f "${RUNDIR}"/*.pid 2>/dev/null || true

  msg "${LOGTAG} all stop commands issued."
}

main "$@"
