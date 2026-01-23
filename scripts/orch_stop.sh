#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUNDIR="run"
LOGTAG="[STOP]"

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

stop_pid() {
    local name="$1"
    local pid="$2"

    if [[ -z "${pid:-}" ]]; then
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "${LOGTAG} ${name} pid=${pid} not running"
        return 0
    fi

    echo "${LOGTAG} ${name} pid=${pid} (TERM)"
    kill -TERM "$pid" 2>/dev/null || true

    local i
    for i in 1 2 3 4 5; do
        sleep 0.2
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "${LOGTAG} ${name} pid=${pid} stopped"
            return 0
        fi
    done

    echo "${LOGTAG} ${name} pid=${pid} still alive (KILL)"
    kill -KILL "$pid" 2>/dev/null || true
    sleep 0.1
    return 0
}

stop_from_pidfiles() {
    shopt -s nullglob

    # 1) scanner_w* pidfile'ları
    for pidfile in "${RUNDIR}"/scanner_*.pid; do
        local name pid
        name="$(basename "$pidfile" .pid)"
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        stop_pid "$name" "$pid"
        rm -f "$pidfile" || true
    done

    # 2) tekil pidfile'lar
    for name in intent_bridge master_executor top_selector aggregator; do
        local pidfile pid
        pidfile="${RUNDIR}/${name}.pid"
        if [[ -f "$pidfile" ]]; then
            pid="$(cat "$pidfile" 2>/dev/null || true)"
            stop_pid "$name" "$pid"
            rm -f "$pidfile" || true
        else
            echo "${LOGTAG} ${name} pidfile missing"
        fi
    done

    shopt -u nullglob
}

stop_from_pgrep() {
    # pidfile kaçarsa diye “ikinci güvenlik ağı”
    # Not: pkill -f yerine pid listesiyle stop_pid kullanalım (daha kontrollü)

    local -a patterns=(
        "orchestration/scanners/worker_stub.py"
        "orchestration/aggregator/run_aggregator.py"
        "Aggregator\\(RedisBus\\(\\)\\)\\.run_forever"
        "orchestration/selector/top_selector.py"
        "orchestration/executor/master_executor.py"
        "orchestration/executor/intent_bridge.py"
    )

    local pat
    for pat in "${patterns[@]}"; do
        # shellcheck disable=SC2207
        local pids=($(pgrep -af "$pat" | awk '{print $1}' | sort -u || true))
        if [[ "${#pids[@]}" -eq 0 ]]; then
            continue
        fi
        local pid
        for pid in "${pids[@]}"; do
            stop_pid "pgrep:${pat}" "$pid"
        done
    done
}

main() {
    stop_from_pidfiles
    stop_from_pgrep

    # Son temizlik: pidfile bırakma
    rm -f "${RUNDIR}"/*.pid 2>/dev/null || true
    echo "${LOGTAG} all stop commands issued."
}

main "$@"
