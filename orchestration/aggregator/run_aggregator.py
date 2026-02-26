from __future__ import annotations

import os
import signal
import sys
import time
from typing import Optional

from orchestration.event_bus.redis_bus import RedisBus
from orchestration.aggregator.aggregator import Aggregator


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _try_load_env() -> None:
    """
    Optional: load .env (if present) using scripts/load_env.sh
    Manuel run'larda hayat kurtarır.
    """
    if not os.path.exists(".env"):
        return
    loader = os.path.join("scripts", "load_env.sh")
    if os.path.exists(loader) and os.access(loader, os.X_OK):
        os.system(f'./{loader} .env >/dev/null 2>&1 || true')


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False
        self.signal_name: Optional[str] = None

    def handler(self, signum: int, _frame) -> None:
        self.requested = True
        try:
            self.signal_name = signal.Signals(signum).name
        except Exception:
            self.signal_name = str(signum)
def main() -> None:
    _try_load_env()

    crash_retry = _env_bool("AGG_CRASH_RETRY", True)
    retry_max_sleep = _env_int("AGG_RETRY_MAX_SLEEP", 30)
    startup_sleep = _env_int("AGG_STARTUP_SLEEP", 0)

    if startup_sleep > 0:
        time.sleep(max(0, startup_sleep))

    sd = _ShutdownFlag()
    signal.signal(signal.SIGINT, sd.handler)
    signal.signal(signal.SIGTERM, sd.handler)

    bus = RedisBus()
    assert bus.ping(), "[AggregatorRunner] Redis ping failed."

    print(
        "[AggregatorRunner] starting... "
        f"signals_stream={getattr(bus, 'signals_stream', 'signals_stream')} "
        f"candidates_stream={getattr(bus, 'candidates_stream', 'candidates_stream')} "
        f"crash_retry={crash_retry} retry_max_sleep={retry_max_sleep}s",
        flush=True,
    )

    sleep_s = 1
    while True:
        if sd.requested:
            print(f"[AggregatorRunner] shutdown requested ({sd.signal_name}). exiting.", flush=True)
            return

        try:
            Aggregator(bus).run_forever()
            print("[AggregatorRunner] run_forever() returned normally. exiting.", flush=True)
            return
        except KeyboardInterrupt:
            print("[AggregatorRunner] KeyboardInterrupt. exiting.", flush=True)
            return
        except Exception as e:
            print(f"[AggregatorRunner][ERROR] crashed: {e!r}", file=sys.stderr, flush=True)
            if not crash_retry:
                raise
            time.sleep(sleep_s)
            sleep_s = min(retry_max_sleep, max(1, sleep_s * 2))


if __name__ == "__main__":
    main()
