import json
from pathlib import Path

models_dir = Path("models")
seq_len_default = 32

for meta_path in models_dir.glob("model_meta_*.json"):
    data = json.loads(meta_path.read_text())
    interval = data.get("interval", meta_path.stem.replace("model_meta_", ""))

    # LSTM dosyaları gerçekten var mı?
    long_path = models_dir / f"lstm_long_{interval}.h5"
    short_path = models_dir / f"lstm_short_{interval}.h5"
    scaler_path = models_dir / f"lstm_scaler_{interval}.joblib"

    has_lstm = long_path.exists() and short_path.exists() and scaler_path.exists()

    # LSTM dosyaları yoksa bu interval için açmayalım
    data["use_lstm_hybrid"] = bool(has_lstm)
    data.setdefault("seq_len", seq_len_default)

    print(f"[META] {meta_path.name}: use_lstm_hybrid={data['use_lstm_hybrid']} (has_lstm={has_lstm})")

    meta_path.write_text(json.dumps(data, indent=2))

print("LSTM hybrid meta update completed.")
