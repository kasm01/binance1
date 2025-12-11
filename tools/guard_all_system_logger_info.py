from pathlib import Path

path = Path("main.py")
text = path.read_text()

lines = text.splitlines(keepends=True)
new_lines = []
wrapped_count = 0

for i, line in enumerate(lines):
    stripped = line.lstrip()
    # Sadece yorum olmayan ve system_logger.info içeren satırlar
    if "system_logger.info(" in line and not stripped.startswith("#"):
        # Bir önceki anlamlı satır "if system_logger" ise tekrar sarmayalım
        j = len(new_lines) - 1
        prev_nonempty = ""
        while j >= 0:
            pl = new_lines[j].strip()
            if pl != "":
                prev_nonempty = pl
                break
            j -= 1

        if prev_nonempty.startswith("if system_logger"):
            # Zaten guard var, olduğu gibi ekle
            new_lines.append(line)
            continue

        indent = line[: len(line) - len(stripped)]
        # Guard + içeriği bir seviye kaydır
        new_lines.append(f"{indent}if system_logger:\n")
        new_lines.append(f"{indent}    {stripped}")
        wrapped_count += 1
    else:
        new_lines.append(line)

new_text = "".join(new_lines)
path.write_text(new_text)
print(f"[OK] {wrapped_count} adet system_logger.info satırı guard ile sarıldı.")
