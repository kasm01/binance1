from binance import AsyncClient

async def fetch_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = 500,
    client: AsyncClient | None = None,
):
    """
    Binance futures için kline (OHLCV) datası çeker.
    """
    if client is None:
        raise ValueError("fetch_klines için 'client' parametresi zorunlu.")

    klines = await client.futures_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )
    return klines
