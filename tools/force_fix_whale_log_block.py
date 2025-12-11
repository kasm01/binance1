from pathlib import Path

path = Path("main.py")
text = path.read_text()

marker = "whale_detector = MultiTimeframeWhaleDetector()"

if marker not in text:
    print("[ERROR] Marker bulunamadı:", marker)
    raise SystemExit(1)

lines = text.splitlines()

# 1) İçinde 'MultiTimeframeWhaleDetector başarıyla init edildi' geçen satırları yorum satırı yap
fixed_lines = []
for line in lines:
    if "MultiTimeframeWhaleDetector başarıyla init edildi" in line:
        if not line.lstrip().startswith("#"):
            fixed_lines.append("# " + line)
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

text = "\n".join(fixed_lines)

# 2) whale_detector satırını bul ve altına temiz blok ekle
idx = text.index(marker)
before = text[:idx]
after = text[idx:]

# whale_detector satırının sonuna kadar git
newline_pos = after.find("\n")
if newline_pos == -1:
    whale_line = after
    rest = ""
else:
    whale_line = after[:newline_pos+1]
    rest = after[newline_pos+1:]

block = (
    f"{whale_line}"
    "try:\n"
    "    if system_logger:\n"
    '        system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")\n'
    "except Exception:\n"
    "    # logger yoksa veya hata olursa sessiz geç\n"
    "    pass\n"
)

new_text = before + block + rest
path.write_text(new_text)
print("[OK] Whale log bloğu zorla düzeltildi.")
