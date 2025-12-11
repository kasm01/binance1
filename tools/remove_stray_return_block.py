from pathlib import Path

path = Path("main.py")
text = path.read_text()
lines = text.splitlines()

out_lines = []
skipping = False
removed = False

for line in lines:
    stripped = line.strip()

    # return { ... } bloğunu başlatan satır
    if not skipping and stripped.startswith("return {"):
        skipping = True
        removed = True
        continue

    # Bloğun kapanışı: tek başına "}" satırı
    if skipping:
        if stripped == "}":
            # bu satırı da atla ve blok sonlandır
            skipping = False
            continue
        # blok içindeki diğer satırları da atlıyoruz
        continue

    out_lines.append(line)

if not removed:
    print("Uyarı: return {...} bloğu bulunamadı, dosya değişmedi.")
else:
    backup_path = path.with_suffix(".py.bak_stray_return")
    backup_path.write_text(text)
    path.write_text("\n".join(out_lines))
    print(f"Stray return bloğu kaldırıldı. Yedek: {backup_path}")
