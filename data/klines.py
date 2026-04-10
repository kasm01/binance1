import asyncio
import time
from typing import Dict, Tuple, List, Any

from binance import AsyncClient


# son başarılı kline cache
_kline_cache: Dict[Tuple[str, str, int], List[Any]] = {}


async def fetch_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = 500,
    client: AsyncClient | None = None,
):
    """
    Binance Futures OHLCV verisi çeker.

    Özellikler:
    - timeout koruması
    - retry mekanizması
    - cache fallback
    - latency ölçümü
    """

    if client is None:
        raise ValueError("fetch_klines için 'client' parametresi zorunlu.")

    cache_key = (symbol, interval, limit)

    timeout_sec = 1.5
    retries = 2

    for attempt in range(retries + 1):

        start_t = time.time()

        try:
            klines = await asyncio.wait_for(
                client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                ),
                timeout=timeout_sec,
            )

            latency = time.time() - start_t

            # cache güncelle
            _kline_cache[cache_key] = klines

            # latency çok yüksekse uyarı
            if latency > 2.0:
                print(
                    f"[DATA][WARN] slow fetch_klines | "
                    f"{symbol} {interval} latency={latency:.2f}s"
                )

            return klines

        except asyncio.TimeoutError:

            if attempt >= retries:

                # cache fallback
                if cache_key in _kline_cache:
                    print(
                        f"[DATA][CACHE] using cached klines | "
                        f"{symbol} {interval}"
                    )
                    return _kline_cache[cache_key]

                raise

            await asyncio.sleep(0.2)

        except Exception:

            if attempt >= retries:

                if cache_key in _kline_cache:
                    print(
                        f"[DATA][CACHE] using cached klines | "
                        f"{symbol} {interval}"
                    )
                    return _kline_cache[cache_key]

                raise

            await asyncio.sleep(0.2)
