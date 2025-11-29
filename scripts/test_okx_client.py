import os
from dotenv import load_dotenv

from core.okx_client import OkxClient


def main():
    # .env yükle (lokal test için)
    load_dotenv(".env")

    print("=== OKX CLIENT TEST ===")

    # Client oluştur
    client = OkxClient()

    # 1) USDT bakiyesi
    balance = client.get_usdt_balance()
    print(f"USDT balance: {balance}")

    # 2) Son 10 mum (BTC-USDT-SWAP, 1m)
    klines = client.get_klines(inst_id="BTC-USDT-SWAP", bar="1m", limit=10)
    print(f"Klines length: {len(klines)}")
    if klines:
        print("First kline:", klines[0])
        print("Last  kline:", klines[-1])

    print("OKX test finished successfully.")


if __name__ == "__main__":
    main()
