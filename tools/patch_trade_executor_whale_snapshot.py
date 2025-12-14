from __future__ import annotations
from pathlib import Path
import re
import sys

FILE = Path("core/trade_executor.py")

MARK_OPEN = "# [BT-WHALE-SNAPSHOT] attach whale snapshot into position meta"
MARK_CLOSE = "# [BT-WHALE-SNAPSHOT] copy whale snapshot from meta into closed dict"

def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)

def already_patched(text: str) -> bool:
    return (MARK_OPEN in text) or (MARK_CLOSE in text)

def patch_execute_decision(text: str) -> str:
    """
    execute_decision içinde yeni pozisyon açılırken meta dict’i oluşturulduğu yere
    whale snapshot kopyalayan bir blok ekler.
    En stabil anchor: meta içinde 'opened_at' ve 'source' geçmesi (loglarda da var).
    """
    if MARK_OPEN in text:
        return text

    # meta dict başlangıcını bul: meta = {... 'opened_at': ..., 'source': ... }
    # birkaç farklı format yakalayalım.
    patterns = [
        r"(?P<head>\n[ \t]*meta\s*=\s*\{\s*\n)(?P<body>.*?\n[ \t]*\}\s*)",
        r"(?P<head>\n[ \t]*meta\s*=\s*dict\(\s*\n)(?P<body>.*?\n[ \t]*\)\s*)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL)
        if not m:
            continue

        block = m.group(0)

        # Bu meta bloğu gerçekten position open için mi? opened_at ve source arayalım.
        if ("opened_at" not in block) or ("source" not in block):
            continue

        insert = f"""
{MARK_OPEN}
                # extra içinden whale snapshot (açılış anı) -> position meta'ya yaz
                try:
                    _ex = extra if isinstance(extra, dict) else {{}}
                    meta["whale_dir"] = _ex.get("whale_dir")
                    meta["whale_score"] = _ex.get("whale_score")
                    meta["whale_on"] = _ex.get("whale_on")
                    meta["whale_alignment"] = _ex.get("whale_alignment")
                    meta["whale_thr"] = _ex.get("whale_thr")
                    meta["model_confidence_factor"] = _ex.get("model_confidence_factor")
                except Exception:
                    pass
"""
        # meta bloğunun hemen SONUNA ekle
        patched_block = block + insert
        text2 = text.replace(block, patched_block, 1)
        return text2

    die("[ERR] execute_decision içinde 'meta' bloğu (opened_at/source) bulunamadı. Dosya yapısı farklı olabilir.")

def patch_close_position(text: str) -> str:
    """
    _close_position içinde kapatılan trade dict'i üretilirken
    position.meta içindeki whale snapshot'ı kolonlara kopyalar.
    Anchor: return edilen dict / closed = {...} / trade = {...} benzeri blok.
    """
    if MARK_CLOSE in text:
        return text

    # _close_position fonksiyon bloğunu kabaca yakala
    fm = re.search(r"\n[ \t]*def\s+_close_position\s*\(.*?\):\n", text)
    if not fm:
        die("[ERR] def _close_position bulunamadı.")

    # fonksiyon gövdesinin yaklaşık bitişini bul: bir sonraki "def " veya "async def " ile.
    start = fm.start()
    after = text[fm.end():]
    nm = re.search(r"\n[ \t]*(async\s+def|def)\s+\w+\s*\(", after)
    end = fm.end() + (nm.start() if nm else len(after))
    func = text[start:end]

    # Kapalı trade dict'i genelde "closed = { ... }" veya "trade = { ... }" gibi.
    # O bloğun hemen SONUNA snapshot kopyasını ekleyeceğiz.
    dict_pat = r"(?P<var>\b(closed|closed_trade|trade|result)\b)\s*=\s*\{"
    dm = re.search(dict_pat, func)
    if not dm:
        die("[ERR] _close_position içinde kapalı trade dict'i atan (closed= {...}) yapı bulunamadı.")

    var = dm.group("var")

    # dict bloğunun kapanışını (} ) bulmak zor; en güvenlisi return öncesine eklemek.
    # return satırını yakalayıp onun hemen ÖNCESİNE ekleyelim.
    rm = re.search(r"\n([ \t]*)return\s+" + re.escape(var) + r"\b", func)
    if not rm:
        # return var yoksa, herhangi bir return öncesine ekleyelim
        rm = re.search(r"\n([ \t]*)return\b", func)
        if not rm:
            die("[ERR] _close_position içinde return bulunamadı.")

    indent = rm.group(1)

    insert = f"""
{indent}{MARK_CLOSE}
{indent}# position.meta -> closed dict'e whale snapshot kopyala (açılış anı doğru kalır)
{indent}try:
{indent}    _m = None
{indent}    # PositionManager/TradeExecutor farklı yapılar olabilir; en yaygın: self.position_manager.current_position.meta
{indent}    try:
{indent}        _pos = getattr(self, "position_manager", None)
{indent}        _cp = getattr(_pos, "current_position", None) if _pos is not None else None
{indent}        _m = getattr(_cp, "meta", None) if _cp is not None else None
{indent}    except Exception:
{indent}        _m = None
{indent}
{indent}    if isinstance(_m, dict):
{indent}        {var}["whale_dir"] = _m.get("whale_dir", {var}.get("whale_dir"))
{indent}        {var}["whale_score"] = _m.get("whale_score", {var}.get("whale_score"))
{indent}        {var}["whale_on"] = _m.get("whale_on", {var}.get("whale_on"))
{indent}        {var}["whale_alignment"] = _m.get("whale_alignment", {var}.get("whale_alignment"))
{indent}        {var}["whale_thr"] = _m.get("whale_thr", {var}.get("whale_thr"))
{indent}        {var}["model_confidence_factor"] = _m.get("model_confidence_factor", {var}.get("model_confidence_factor"))
{indent}except Exception:
{indent}    pass
"""

    func2 = func[:rm.start()] + insert + func[rm.start():]
    return text[:start] + func2 + text[end:]

def main() -> None:
    if not FILE.exists():
        die(f"[ERR] File not found: {FILE}")

    text = FILE.read_text(encoding="utf-8")
    if already_patched(text):
        print("[OK] Zaten patch'li görünüyor, işlem yok.")
        return

    text = patch_execute_decision(text)
    text = patch_close_position(text)

    FILE.write_text(text, encoding="utf-8")
    print(f"[OK] Patch uygulandı: {FILE}")

if __name__ == "__main__":
    main()
