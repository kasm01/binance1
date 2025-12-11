import re
from pathlib import Path

path = Path("main.py")
text = path.read_text()

# HybridModel, MTFModel, WhaleDetector ve TradeExecutor bloklarının başlıkları
patterns = [
    r"hybrid_model\s*=",
    r"HybridMultiTFModel",
    r"MultiTimeframeWhaleDetector",
    r"TradeExecutor\("
]

# Eğer bu bloklar bootstrap() dışında başlıyorsa uyarı ver
for pat in patterns:
    if re.search(pat, text) and not re.search(r"def bootstrap", text):
        print(f"[WARN] Pattern appears outside bootstrap(): {pat}")

# Sadece kullanıcıya rehberlik scripti — auto-fix yapmadık
print("\n⚠ Bu blokların tamamı def bootstrap(): altında olmalı.\n"
      "Eğer istersen auto-fix eden sürümü gönderirim.\n")

