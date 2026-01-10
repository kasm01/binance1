import json
from pathlib import Path
import joblib
import tensorflow as tf

INTERVALS = ["1m","3m","5m","15m","30m","1h"]

def main():
    models_dir = Path("models")
    ok = True
    for itv in INTERVALS:
        meta_p = models_dir / f"model_meta_{itv}.json"
        scaler_p = models_dir / f"lstm_scaler_{itv}.joblib"
        h5_p = models_dir / f"lstm_long_{itv}.h5"

        print(f"\n== {itv} ==")

        if not meta_p.exists():
            print("META MISSING", meta_p)
            ok = False
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        feat_n = len(meta.get("feature_schema", []) or meta.get("feature_cols", []) or [])
        print("meta feature_n =", feat_n)

        if not scaler_p.exists():
            print("SCALER MISSING", scaler_p)
            ok = False
        else:
            sc = joblib.load(scaler_p)
            sn = getattr(sc, "n_features_in_", None)
            print("scaler n_features_in_ =", sn)
            if feat_n and sn and feat_n != sn:
                print("!! MISMATCH meta vs scaler")
                ok = False

        if not h5_p.exists():
            print("MODEL MISSING", h5_p)
            ok = False
        else:
            m = tf.keras.models.load_model(str(h5_p))
            ishape = m.input_shape
            print("model input_shape =", ishape)
            # (None, window, n_features)
            if isinstance(ishape, tuple) and len(ishape) == 3 and feat_n:
                mn = ishape[2]
                if mn != feat_n:
                    print("!! MISMATCH meta vs model n_features")
                    ok = False

    print("\nRESULT:", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 2)

if __name__ == "__main__":
    main()
