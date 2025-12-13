from pathlib import Path
import re
import sys

path = Path("backtest_mtf.py")
txt = path.read_text(encoding="utf-8")

marker = "notional = notional * aggr_factor"
if marker not in txt:
    print("[ERR] notional hesap bloğu bulunamadı")
    sys.exit(1)

block = """
    # ----------------------------------------------------------
    # Whale-based dynamic position sizing
    # ----------------------------------------------------------
    whale_alignment = None
    try:
        whale_alignment = (extra or {}).get("whale_alignment")
    except Exception:
        whale_alignment = None

    size_mode = os.getenv("BT_WHALE_SIZE_MODE", "aligned_only").strip().lower()
    max_boost = float(os.getenv("BT_WHALE_SIZE_MAX_BOOST", "2.0"))
    min_scale = float(os.getenv("BT_WHALE_SIZE_MIN_SCALE", "1.0"))

    try:
        wscore = float(whale_score or 0.0)
    except Exception:
        wscore = 0.0

    # whale_score 0..1 → scale 1..max_boost
    score_scale = 1.0 + (max_boost - 1.0) * max(0.0, min(1.0, wscore))

    whale_scale = 1.0

    if size_mode == "off":
        whale_scale = 1.0

    elif size_mode == "both":
        whale_scale = score_scale

    else:  # aligned_only
        if whale_alignment == "aligned":
            whale_scale = score_scale
        elif whale_alignment == "opposed":
            whale_scale = min_scale
        else:
            whale_scale = 1.0

    notional = notional * whale_scale
"""

txt = txt.replace(marker, marker + block)

path.write_text(txt, encoding="utf-8")
print("[OK] Whale dynamic sizing eklendi")
