import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGET_FILES = [
    ROOT / "training" / "offline_pretrain_six_months.py",
    ROOT / "models" / "train_lstm_for_interval.py",
]

def patch_build_features_block(path: Path) -> None:
    if not path.exists():
        print(f"[SKIP] {path} yok, atlanıyor.")
        return

    text = path.read_text(encoding="utf-8")

    # Zaten alias_map bloğu eklenmiş mi?
    if "alias_map = {" in text and "taker_buy_base_volume" in text:
        print(f"[OK] {path.name}: alias_map bloğu zaten var, dokunulmadı.")
        return

    # Her türlü "<indent><var> = build_features(...)" desenini yakala
    pattern = (
        r"(?P<indent>[ \t]*)"
        r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?P<rhs>build_features\([^)]*\))"
    )

    def repl(m: re.Match) -> str:
        indent = m.group("indent")
        var = m.group("var")
        rhs = m.group("rhs")
        inner = indent + "    "

        alias_block = (
            f"{indent}{var} = {rhs}\n"
            f"{indent}# ------------------------------------------------------------\n"
            f"{indent}# Backward-compat: eski kolon adlarını yeni kolonlardan üret\n"
            f"{indent}# ------------------------------------------------------------\n"
            f"{indent}alias_map = {{\n"
            f'{inner}"taker_buy_base_volume": "taker_buy_base_asset_volume",\n'
            f'{inner}"taker_buy_quote_volume": "taker_buy_quote_asset_volume",\n'
            f"{indent}}}\n"
            f"{indent}for old_col, new_col in alias_map.items():\n"
            f"{inner}if old_col not in {var}.columns and new_col in {var}.columns:\n"
            f"{inner}    {var}[old_col] = {var}[new_col]\n"
        )
        return alias_block

    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        print(
            f"[WARN] {path.name}: 'X = build_features(...)' deseni bulunamadı, "
            "patch uygulanmadı."
        )
        return

    # Yedek al
    backup_path = path.with_suffix(path.suffix + ".bak_auto_patch")
    backup_path.write_text(text, encoding="utf-8")

    path.write_text(new_text, encoding="utf-8")
    print(
        f"[PATCHED] {path.name}: alias_map + backward-compat bloğu eklendi "
        f"(backup: {backup_path.name})."
    )


def main() -> None:
    print("=== LSTM & feature kolon auto-patch scripti (v2) ===")
    for p in TARGET_FILES:
        patch_build_features_block(p)

    # HybridModel LSTM sequence tarafı için sadece bilgi mesajı
    hybrid_path = ROOT / "models" / "hybrid_inference.py"
    if hybrid_path.exists():
        h_text = hybrid_path.read_text(encoding="utf-8")
        if "def _build_lstm_sequences" in h_text and "seq_len" in h_text:
            print(
                f"[INFO] {hybrid_path.name}: _build_lstm_sequences zaten mevcut, "
                "LSTM sequence mantığı korundu."
            )
        else:
            print(
                f"[WARN] {hybrid_path.name}: _build_lstm_sequences beklenen formda değil, "
                "manuel kontrol önerilir."
            )
    else:
        print("[SKIP] models/hybrid_inference.py bulunamadı.")


if __name__ == "__main__":
    main()
