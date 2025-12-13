from __future__ import annotations
from pathlib import Path
import re
import sys

INSERT_BLOCK = r'''
# ----------------------------------------------------------
# Whale policy (opposed veto / only trade aligned)
# ----------------------------------------------------------
WHALE_FILTER = get_bool_env("BT_WHALE_FILTER", False)
WHALE_THR = float(os.getenv("BT_WHALE_THR", "0.50"))
WHALE_VETO_OPPOSED = get_bool_env("BT_WHALE_VETO_OPPOSED", True)

whale_on = (
    whale_score is not None
    and float(whale_score) >= WHALE_THR
    and str(whale_dir).lower() not in ("none", "nan", "null", "")
)

whale_alignment = "no_whale"
if whale_on:
    wd = str(whale_dir).lower().strip()
    if signal in ("long", "short") and wd in ("long", "short"):
        whale_alignment = "aligned" if signal == wd else "opposed"
    else:
        whale_alignment = "other"

if WHALE_FILTER and WHALE_VETO_OPPOSED:
    if whale_alignment == "opposed":
        system_logger.info(
            "[BT-WHALE-POLICY] VETO opposed | bar=%d signal=%s whale_dir=%s whale_score=%.3f thr=%.2f -> HOLD",
            i, signal, str(whale_dir), float(whale_score), WHALE_THR
        )
        signal = "hold"

# extra içine yaz (analiz tarafında da görünsün)
extra["whale_alignment"] = whale_alignment
extra["whale_on"] = bool(whale_on)
extra["whale_thr"] = float(WHALE_THR)
'''

def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("backtest_mtf.py")
    s = path.read_text(encoding="utf-8")

    if "BT-WHALE-POLICY" in s:
        print("[SKIP] whale policy zaten var.")
        return 0

    # Bu sinyal bloğunu hedefliyoruz (senin dosyandakiyle aynı)
    patt = re.compile(
        r'(?ms)^(\s*#\s*signal\s*\n\s*signal\s*=\s*"hold"\s*\n\s*if\s+p_used\s*>=\s*long_thr:\s*\n\s*signal\s*=\s*"long"\s*\n\s*elif\s+p_used\s*<=\s*short_thr:\s*\n\s*signal\s*=\s*"short"\s*\n)'
    )

    m = patt.search(s)
    if not m:
        print("[ERROR] hedef signal bloğu bulunamadı. Dosyadaki '# signal' bölümünü kontrol et.")
        return 2

    insert_at = m.end(1)
    s2 = s[:insert_at] + INSERT_BLOCK + s[insert_at:]

    path.write_text(s2, encoding="utf-8")
    print(f"[OK] eklendi: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
