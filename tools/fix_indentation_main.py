import re
from pathlib import Path

path = Path("main.py")
txt = path.read_text().splitlines()

# create_runtime_context fonksiyonunu bul
start = None
for i, line in enumerate(txt):
    if line.strip().startswith("def create_runtime_context"):
        start = i
        break

if start is None:
    raise SystemExit("Fonksiyon bulunamadı!")

# Fonksiyon gövdesi 4 boşluk ile başlamalı
correct_indent = " " * 4

# Fonksiyon bitene kadar tüm satırları hizala
i = start + 1
while i < len(txt):
    line = txt[i]

    # Fonksiyon bitişi → bir sonraki def
    if line.strip().startswith("def ") or line.strip().startswith("class "):
        break

    if line.strip() == "":
        i += 1
        continue

    # Eğer satır 0 sütundan başlıyorsa → fonksiyon dışına taşmıştır → hizala
    if not line.startswith(correct_indent):
        txt[i] = correct_indent + line.lstrip()

    i += 1

# Dosyayı geri yaz
path.write_text("\n".join(txt) + "\n")
print("✔ Indentation düzeltildi.")
