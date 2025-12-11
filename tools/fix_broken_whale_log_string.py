from pathlib import Path
import re

path = Path("main.py")
text = path.read_text()

# Bozuk logger satırı (parçalanmış string) için geniş regex
pattern = r'system_logger\.info\([^)]*init edildi[^)]*\)'

# Eğer satır tek parça değilse, onu düzgün biçime dönüştüreceğiz
if "init edildi" in text:
    print("[INFO] init edildi içeren satır bulundu, düzeltme deneniyor...")

    # Düzgün olması gereken satır
    correct = (
        'try:\n'
        '    if system_logger:\n'
        '        system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")\n'
        'except Exception:\n'
        '    pass\n'
    )

    # Eski / bozuk logger satırlarını temizle
    new_text = re.sub(
        r'try:\s*\n(?:.*\n){0,5}?system_logger\.info\([^)]*\)\s*\n(?:.*\n){0,3}?except.*?pass',
        correct,
        text,
        flags=re.DOTALL
    )

    # Ayrıca tek başına kırık logger satırlarını temizle
    new_text = re.sub(
        r'system_logger\.info\([^)]*init edildi[^)]*\)',
        'system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")',
        new_text
    )

    path.write_text(new_text)
    print("[OK] Broken logger string fixed.")
else:
    print("[WARN] init edildi satırı bulunamadı.")
