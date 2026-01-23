from __future__ import annotations

from orchestration.event_bus.redis_bus import RedisBus
from orchestration.aggregator.aggregator import Aggregator


def main() -> None:
    Aggregator(RedisBus()).run_forever()


if __name__ == "__main__":
    main()
