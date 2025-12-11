from pathlib import Path
import re

path = Path("main.py")
text = path.read_text()

pattern = r'(?P<indent>\s*)system_logger\.info\("\[WHALE\] MultiTimeframeWhaleDetector başarıyla init edildi\."\)'
replacement = (
    r'\g<indent>if system_logger:\n'
    r'\g<indent>    system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")'
)

new_text, n = re.subn(pattern, replacement, text, count=1)
if n == 0:
    print("[ERR] Hedef logger satırı bulunamadı, hiçbir değişiklik yapılmadı.")
else:
    path.write_text(new_text)
    print(f"[OK] {n} adet whale logger satırı korumalı hale getirildi.")
