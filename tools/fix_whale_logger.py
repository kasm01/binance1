from pathlib import Path

path = Path("main.py")
text = path.read_text()

old = 'system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")'

if old not in text:
    print("[INFO] Hedef log satırı bulunamadı, değişiklik yapılmadı.")
    raise SystemExit(0)

new = """try:
    if system_logger:
        system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")
except Exception:
    # Logger henüz init edilmemiş olabilir; sessizce geç
    pass"""

text = text.replace(old, new, 1)
path.write_text(text)

print("[OK] WhaleDetector init log satırı güvenli hale getirildi.")
