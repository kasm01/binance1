from __future__ import annotations
from pathlib import Path
import re
import sys

POLICY_START = r"# ----------------------------------------------------------\n# Whale policy (opposed veto / only trade aligned)\n# ----------------------------------------------------------"
POLICY_END_MARK = r'extra\["whale_thr"\]\s*=\s*float\(WHALE_THR\)\s*'

def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("backtest_mtf.py")
    s = path.read_text(encoding="utf-8")

    # 1) policy bloğunu yakala
    m = re.search(
        rf"(?ms)^{re.escape(POLICY_START)}.*?^{POLICY_END_MARK}.*?$",
        s,
    )
    if not m:
        print("[ERROR] Whale policy bloğu bulunamadı. (BT-WHALE-POLICY içermiyor olabilir)")
        return 2

    policy_block = s[m.start():m.end()]
    # policy içindeki satırları indent’siz normalize et (sonra hedef indent ile tekrar basacağız)
    policy_lines = policy_block.splitlines()
    # Baştaki 0-indented satırları koru, ama önce tüm satırlardan ortak leading boşluğu kırp
    # (policy zaten 0-indent eklendiyse bu no-op)
    # Ortak minimum indent:
    def leading_spaces(line: str) -> int:
        if not line.strip():
            return 10**9
        return len(line) - len(line.lstrip(" "))
    min_indent = min((leading_spaces(l) for l in policy_lines), default=0)
    if min_indent == 10**9:
        min_indent = 0
    policy_norm = "\n".join([l[min_indent:] if len(l) >= min_indent else l for l in policy_lines]).rstrip() + "\n"

    # 2) policy bloğunu dosyadan kaldır
    s_wo = s[:m.start()] + s[m.end():]

    # 3) hedef insertion noktası: signal üretim bloğunun hemen sonrası (try içindeki indent ile)
    # Senin dosyada bu kalıp var:
    # signal = "hold"
    # if p_used >= long_thr: ...
    sig = re.search(
        r'(?ms)^(?P<ind>\s*)signal\s*=\s*"hold"\s*\n(?P=ind)if\s+p_used\s*>=\s*long_thr:\s*\n(?P=ind)\s*signal\s*=\s*"long"\s*\n(?P=ind)elif\s+p_used\s*<=\s*short_thr:\s*\n(?P=ind)\s*signal\s*=\s*"short"\s*\n',
        s_wo
    )
    if not sig:
        print("[ERROR] Signal bloğu bulunamadı. backtest_mtf.py içinde signal üretim kısmını kontrol et.")
        return 3

    indent = sig.group("ind")
    insert_at = sig.end()

    # policy’yi hedef indent ile yeniden yaz
    policy_reindented = "\n".join([(indent + line) if line.strip() else line for line in policy_norm.splitlines()]).rstrip() + "\n"

    s_fixed = s_wo[:insert_at] + policy_reindented + s_wo[insert_at:]

    path.write_text(s_fixed, encoding="utf-8")
    print(f"[OK] whale policy taşındı + indent düzeltildi: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
