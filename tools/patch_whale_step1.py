from __future__ import annotations
import re
import sys
from pathlib import Path

BLOCK_MARK = "[BT-WHALE-STEP1]"
def die(msg: str, code: int = 1):
    print(f"[ERR] {msg}")
    sys.exit(code)

def main():
    if len(sys.argv) < 2:
        die("Kullanım: python tools/patch_whale_step1.py backtest_mtf.py")

    path = Path(sys.argv[1])
    s = path.read_text(encoding="utf-8")

    # 0) Zaten eklendiyse tekrar ekleme (idempotent)
    if BLOCK_MARK in s:
        print(f"[OK] Zaten patchli: {path}")
        return

    # 1) Whale policy bloğunu, loop içinde '# ATR' satırından hemen önce ekle
    #    (whale_dir/whale_score hesaplandıktan sonra olmalı — kodunuzda ATR genelde onun altında)
    m = re.search(r"\n(?P<indent>\s*)#\s*ATR\b", s)
    if not m:
        die("'# ATR' marker bulunamadı. backtest_mtf.py içinde ATR bloğunun comment satırını kontrol et.")

    indent = m.group("indent")

    insert_block = f"""
{indent}# ----------------------------------------------------------
{indent}# Whale policy + whale-only + notional scaling  {BLOCK_MARK}
{indent}# ----------------------------------------------------------
{indent}WHALE_FILTER = get_bool_env("BT_WHALE_FILTER", False)
{indent}WHALE_ONLY = get_bool_env("BT_WHALE_ONLY", False)
{indent}WHALE_THR = float(os.getenv("BT_WHALE_THR", "0.50"))
{indent}WHALE_VETO_OPPOSED = get_bool_env("BT_WHALE_VETO_OPPOSED", False)
{indent}
{indent}# opposed durumda notional'ı küçültmek için model_confidence_factor'ı düşüreceğiz
{indent}OPPOSED_SCALE = float(os.getenv("BT_WHALE_OPPOSED_SCALE", "0.30"))   # 0.30 => %70 küçült
{indent}ALIGNED_BOOST  = float(os.getenv("BT_WHALE_ALIGNED_BOOST", "1.00"))  # şimdilik boost yok
{indent}
{indent}whale_on = (
{indent}    whale_score is not None
{indent}    and float(whale_score) >= WHALE_THR
{indent}    and str(whale_dir).lower() not in ("none", "nan", "null", "")
{indent})
{indent}
{indent}whale_alignment = "no_whale"
{indent}if whale_on:
{indent}    wd = str(whale_dir).lower().strip()
{indent}    if signal in ("long", "short") and wd in ("long", "short"):
{indent}        whale_alignment = "aligned" if signal == wd else "opposed"
{indent}    else:
{indent}        whale_alignment = "other"
{indent}
{indent}# Whale-only: whale_on değilse veya aligned değilse trade açma
{indent}if WHALE_ONLY:
{indent}    if (not whale_on) or (whale_alignment != "aligned"):
{indent}        system_logger.info(
{indent}            "[BT-WHALE-ONLY] HOLD | bar=%d signal=%s whale_dir=%s whale_score=%.3f thr=%.2f alignment=%s",
{indent}            i, signal, str(whale_dir), float(whale_score or 0.0), WHALE_THR, whale_alignment
{indent}        )
{indent}        signal = "hold"
{indent}
{indent}# Opsiyonel veto: whale_on + opposed ise HOLD
{indent}if WHALE_FILTER and WHALE_VETO_OPPOSED and whale_alignment == "opposed":
{indent}    system_logger.info(
{indent}        "[BT-WHALE-VETO] HOLD(opposed) | bar=%d signal=%s whale_dir=%s whale_score=%.3f thr=%.2f",
{indent}        i, signal, str(whale_dir), float(whale_score or 0.0), WHALE_THR
{indent}    )
{indent}    signal = "hold"
{indent}
{indent}# Notional scaling: TradeExecutor _compute_notional model_confidence_factor kullanıyor.
{indent}model_confidence_factor = 1.0
{indent}if whale_on and whale_alignment == "aligned":
{indent}    model_confidence_factor *= ALIGNED_BOOST
{indent}elif whale_on and whale_alignment == "opposed":
{indent}    model_confidence_factor *= OPPOSED_SCALE
"""

    s2 = s[: m.start()] + insert_block + s[m.start():]

    # 2) bt_context.update içine whale_on / whale_alignment / whale_thr / model_conf ekle
    #    whale_score satırının altına yerleştir.
    #    (bt_context patch'i yoksa da sorun değil; ekleriz)
    def add_into_bt_context(text: str) -> str:
        # bt_context.update({ ... "whale_score": ..., ... })
        patt = r'(\n\s*"whale_score"\s*:\s*float\(whale_score\)\s*,)'
        mm = re.search(patt, text)
        if not mm:
            # bazı varyantlarda tek tırnak olabilir
            patt2 = r"(\n\s*'whale_score'\s*:\s*float\(whale_score\)\s*,)"
            mm = re.search(patt2, text)
        if not mm:
            return text  # bt_context yoksa pas geç

        ins = (
            mm.group(1)
            + "\n                    \"whale_on\": bool(whale_on),"
            + "\n                    \"whale_alignment\": whale_alignment,"
            + "\n                    \"whale_thr\": float(WHALE_THR),"
            + "\n                    \"model_confidence_factor\": float(model_confidence_factor),"
        )
        return text[: mm.start()] + ins + text[mm.end():]

    s3 = add_into_bt_context(s2)

    # 3) execute_decision extra dict literaline de ekle (analiz + log için)
    def add_into_extra_literal(text: str) -> str:
        # extra={ ... "whale_score": ..., "atr": ... }
        patt = r'(\n\s*"whale_score"\s*:\s*float\(whale_score\)\s*,)'
        mm = re.search(patt, text)
        if not mm:
            patt2 = r"(\n\s*'whale_score'\s*:\s*float\(whale_score\)\s*,)"
            mm = re.search(patt2, text)
        if not mm:
            return text

        ins = (
            mm.group(1)
            + "\n                    \"whale_on\": bool(whale_on),"
            + "\n                    \"whale_alignment\": whale_alignment,"
            + "\n                    \"whale_thr\": float(WHALE_THR),"
            + "\n                    \"model_confidence_factor\": float(model_confidence_factor),"
        )
        return text[: mm.start()] + ins + text[mm.end():]

    s4 = add_into_extra_literal(s3)

    path.write_text(s4, encoding="utf-8")
    print(f"[OK] Step-1 whale patch eklendi: {path}")

if __name__ == "__main__":
    main()
