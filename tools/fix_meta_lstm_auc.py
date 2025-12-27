#!/usr/bin/env python3
import json, re, glob, os, shutil
from pathlib import Path
from datetime import datetime

INTERVALS = ["1m", "5m", "15m", "1h"]

LOG_DIR = Path("logs/training")
MODELS_DIR = Path("models")

# örnek satırlar:
# Epoch 5: val_auc improved from 0.49506 to 0.50544, saving model to ...
# veya: val_auc improved from -inf to 0.49192
VAL_RE = re.compile(r"val_auc improved from [^ ]+ to ([0-9]*\.?[0-9]+)")

def latest_auc_for_interval(interval: str) -> float | None:
    # log isimleri sende farklı varyantlarda (lstm_BTCUSDT_1m_..., lstm_BTCUSDT_1m_h1_w50_..., vs.)
    patterns = [
        str(LOG_DIR / f"lstm_*_{interval}_*.log"),
        str(LOG_DIR / f"lstm_*_{interval}_h*_*.log"),
        str(LOG_DIR / f"lstm_*_{interval}*.log"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        return None

    # en güncel logları önce dene (mtime ile)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for fp in files[:20]:  # çok log varsa ilk 20 yeter
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            matches = VAL_RE.findall(txt)
            if matches:
                # en son improvement en iyi checkpoint genelde
                return float(matches[-1])
        except Exception:
            continue
    return None

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("models/_meta_backup") / ts
    backup_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    for interval in INTERVALS:
        meta_path = MODELS_DIR / f"model_meta_{interval}.json"
        if not meta_path.exists():
            print(f"[SKIP] meta yok: {meta_path}")
            continue

        auc = latest_auc_for_interval(interval)
        if auc is None:
            print(f"[WARN] {interval}: loglardan val_auc bulunamadı -> meta değişmedi")
            continue

        # backup
        shutil.copy2(meta_path, backup_dir / meta_path.name)

        meta = load_json(meta_path)
        meta["use_lstm_hybrid"] = True
        # “long/short” ayrımı sende şimdilik aynı model/kopya olduğu için aynı yazıyoruz
        meta["lstm_long_auc"] = float(auc)
        meta["lstm_short_auc"] = float(auc)

        # seq_len meta’da zaten 50 görünüyor; yoksa dokunma ama varsa int'e çevir
        if "seq_len" in meta:
            try:
                meta["seq_len"] = int(meta["seq_len"])
            except Exception:
                pass

        save_json(meta_path, meta)
        updated += 1
        print(f"[OK] {interval}: lstm_long_auc=lstm_short_auc={auc:.5f} -> {meta_path}")

    print(f"\nDONE. updated={updated} backup_dir={backup_dir}")

if __name__ == "__main__":
    main()
