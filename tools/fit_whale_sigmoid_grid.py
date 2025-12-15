import argparse, itertools, math
import pandas as pd

def sigmoid(x, k, x0):
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))

def pnl_sign(df):
    # realized_pnl varsa onu kullan
    if "realized_pnl" in df.columns:
        return df["realized_pnl"].astype(float)
    # yoksa close-entry*dir
    side = df["side"].astype(str).str.lower()
    entry = df["entry_price"].astype(float)
    close = df["close_price"].astype(float)
    pnl = (close-entry)
    pnl = pnl.where(side=="long", -pnl)
    return pnl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", default="outputs/whale_sigmoid_fit.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.trades)
    if "whale_score" not in df.columns:
        raise SystemExit("whale_score col yok")
    if "whale_alignment" not in df.columns:
        # alignment yoksa direction ile side’dan çıkar
        pass

    ws = df["whale_score"].fillna(0.0).astype(float).clip(0.0, 1.0)
    pnl = pnl_sign(df)

    # objective: whale_factor uygulanmış notional ile pnl'yi maximize et (proxy)
    # burada sadece "scale" etkisini taklit ediyoruz: pnl_scaled = pnl * whale_factor
    # (gerçekte sl/tp etkisi var ama ilk yaklaşım için iyi)
    grid_k   = [2.0, 4.0, 6.0, 8.0, 10.0]
    grid_x0  = [0.30, 0.40, 0.50, 0.60, 0.70]
    grid_min = [0.20, 0.30, 0.40]
    grid_max = [1.20, 1.50, 2.00]

    best = None
    rows = []
    for k, x0, fmin, fmax in itertools.product(grid_k, grid_x0, grid_min, grid_max):
        if fmax <= fmin: 
            continue
        s = ws.map(lambda x: sigmoid(x, k, x0))
        factor = fmin + (fmax - fmin) * s

        # zıt yön cezası: whale_alignment varsa kullan
        if "whale_alignment" in df.columns:
            aligned = df["whale_alignment"].astype(str).str.lower().eq("aligned")
            opposed = df["whale_alignment"].astype(str).str.lower().eq("opposed")
            factor = factor.where(~opposed, (1.0 / factor.clip(1e-6)).clip(lower=fmin, upper=fmax))

        pnl_scaled = pnl * factor
        score = float(pnl_scaled.sum())  # toplam pnl proxy
        rows.append((score, k, x0, fmin, fmax))

    rows.sort(reverse=True, key=lambda x: x[0])
    out = pd.DataFrame(rows, columns=["score_sum_pnl_scaled","k","x0","fmin","fmax"])
    out.to_csv(args.out, index=False)
    print("[OK] saved", args.out)
    print("[BEST]", out.iloc[0].to_dict())

if __name__ == "__main__":
    main()
