from pathlib import Path

path = Path("main.py")
text = path.read_text()

# HATALI blok (bozuk auto-patch'ten gelen)
bad_block_start = 'try:\n    if system_logger:\n        system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")'
# Bazı durumlarda auto-patch indent bozulmuş olabilir; daha geniş pattern yakalayalım
if 'try:' in text and 'MultiTimeframeWhaleDetector başarıyla init edildi' in text:
    # Şimdi bütün bozuk try bloğunu normalize edelim:
    import re

    # try: ile başlayan ve "init edildi." içeren satırı yakala
    pattern = r'try:\s*\n\s*if system_logger:.*?init edildi\.\s*'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if matches:
        print(f"[INFO] Hatalı try bloğu bulundu. {len(matches)} adet")

        correct_block = (
            "try:\n"
            "    if system_logger:\n"
            "        system_logger.info(\"[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.\")\n"
            "except Exception:\n"
            "    pass  # logger init edilmemiş olabilir\n"
        )

        new_text = re.sub(pattern, correct_block, text, flags=re.DOTALL)
        path.write_text(new_text)
        print("[OK] Try/except bloğu düzeltildi.")
    else:
        print("[WARN] Hatalı try bloğu bulunamadı, işlem yapılmadı.")
else:
    print("[WARN] İlgili log satırı main.py içinde bulunamadı.")
