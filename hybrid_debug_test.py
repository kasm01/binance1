import pandas as pd
from models.hybrid_inference import HybridModel


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Basit price/volume feature'ları
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    df["vol_10"] = df["volume"].rolling(window=10, min_periods=1).std()

    # Eğitime uyum için ekstra dummy kolon
    df["dummy_extra"] = 0.0

    # NA'leri temizle (rolling / pct_change sonrası)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    # Offline cache'ten 5m BTC verisini oku
    raw_df = pd.read_csv("data/offline_cache/BTCUSDT_5m_6m.csv")
    raw_df = raw_df.tail(500).reset_index(drop=True)

    print("raw_df shape:", raw_df.shape)
    print("raw_df columns:", raw_df.columns.tolist())

    # Feature'ları üret
    feat_df = build_features(raw_df)
    print("feat_df shape:", feat_df.shape)
    print("feat_df columns:", feat_df.columns.tolist())

    # Hybrid modeli yükle
    hm = HybridModel(model_dir="models", interval="5m")
    p, dbg = hm.predict_proba(feat_df)

    print("\nSon 10 skor:", p[-10:])
    print("\nDebug sözlüğü:")
    for k, v in dbg.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
