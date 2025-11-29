import os
from dotenv import load_dotenv

from binance.client import Client
from binance.exceptions import BinanceAPIException


def main():
    # .env dosyasını yükle
    load_dotenv(".env")

    print("=== BINANCE CLIENT TEST ===")

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("❌ BINANCE_API_KEY / BINANCE_API_SECRET bulunamadı. .env dosyanı kontrol et.")
        return

    client = Client(api_key, api_secret)

    # 1) Ping testi
    try:
        ping = client.ping()
        print("✅ Ping OK:", ping)
    except Exception as e:
        print("❌ Ping hatası:", e)
        return

    # 2) Futures exchange info
    try:
        info = client.futures_exchange_info()
        symbols = info.get("symbols", [])
        print(f"✅ Futures exchange info OK, sembol sayısı: {len(symbols)}")
    except BinanceAPIException as e:
        print("❌ BinanceAPIException (exchange_info):", e)
    except Exception as e:
        print("❌ Beklenmeyen hata (exchange_info):", e)

    # 3) BTCUSDT için son 5 futures kline
    try:
        klines = client.futures_klines(symbol="BTCUSDT", interval="1m", limit=5)
        print(f"✅ Futures klines OK, gelen mum sayısı: {len(klines)}")
        if klines:
            print("  İlk kline:", klines[0])
            print("  Son  kline:", klines[-1])
    except BinanceAPIException as e:
        print("❌ BinanceAPIException (futures_klines):", e)
    except Exception as e:
        print("❌ Beklenmeyen hata (futures_klines):", e)

    # 4) İsterseniz futures_account (IP engelini burada görürüz)
    try:
        account = client.futures_account()
        total_wallet_balance = account.get("totalWalletBalance")
        print(f"✅ futures_account OK, totalWalletBalance={total_wallet_balance}")
    except BinanceAPIException as e:
        print("❌ BinanceAPIException (futures_account):", e)
    except Exception as e:
        print("❌ Beklenmeyen hata (futures_account):", e)

    print("=== BINANCE TEST BİTTİ ===")


if __name__ == "__main__":
    main()
