import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", default="outputs/fit_weight_w_latest.csv")
    ap.add_argument("--long_thr", type=float, default=0.60)
    ap.add_argument("--short_thr", type=float, default=0.40)
    ap.add_argument("--w_min", type=float, default=0.0)
    ap.add_argument("--w_max", type=float, default=1.0)
    ap.add_argument("--w_step", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.trades)

    # gerekli kolonlar
    if "bt_p_used" not in df.columns:
        raise SystemExit("bt_p_used yok")
    if "bt_p_buy_ema" not in df.columns:
        raise SystemExit("bt_p_buy_ema yok (trade row'a eklemen lazım)")

    p_h = df["bt_p_used"].astype(float)
    p_e = df["bt_p_buy_ema"].astype(float)

    # true label: (trade sonucu yön doğru mu?) -> UP=1
    side = df["side"].astype(str).str.lower()
    entry = df["entry_price"].astype(float)
    close = df["close_price"].astype(float)
    true = (((side=="long") & (close>entry)) | ((side=="short") & (close<entry))).astype(int)

    rows=[]
    for w in np.arange(args.w_min, args.w_max + 1e-9, args.w_step):
        p_mix = w*p_e + (1.0-w)*p_h

        pred = np.where(p_mix >= args.long_thr, 1,
               np.where(p_mix <= args.short_thr, 0, -1))

        used = (pred!=-1).sum()
        if used == 0:
            continue

        pred_u = pred[pred!=-1]
        true_u = true[pred!=-1]

        TP = int(((pred_u==1)&(true_u==1)).sum())
        FP = int(((pred_u==1)&(true_u==0)).sum())
        FN = int(((pred_u==0)&(true_u==1)).sum())
        TN = int(((pred_u==0)&(true_u==0)).sum())

        acc = (TP+TN)/used if used else 0
        prec = TP/(TP+FP) if (TP+FP) else 0
        rec = TP/(TP+FN) if (TP+FN) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0

        rows.append([w, used, TP, FP, FN, TN, acc, prec, rec, f1])

    out = pd.DataFrame(rows, columns=["w","used","TP","FP","FN","TN","acc","precision","recall","f1"])
    out = out.sort_values(["f1","acc","used"], ascending=[False, False, False])
    out.to_csv(args.out, index=False)
    print("[OK] saved", args.out)
    print("[BEST]\n", out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
