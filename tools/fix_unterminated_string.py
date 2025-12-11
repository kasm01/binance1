from pathlib import Path
import re

path = Path("main.py")
lines = path.read_text().splitlines()

fixed = []
changes = 0

pattern = re.compile(r'system_logger\.info\(\s*"$')

for line in lines:
    stripped = line.strip()

    # Sistem logger satırı yarım kalmışsa
    if stripped.startswith('system_logger.info("') and stripped.endswith('("'):
        fixed.append("# " + line)
        changes += 1
        continue

    # Tek tırnak açılmış ama kapanmamış olabilir — kaba kontrol:
    if stripped.count('"') == 1 and 'system_logger.info' in stripped:
        fixed.append("# " + line)
        changes += 1
        continue

    fixed.append(line)

path.write_text("\n".join(fixed) + "\n")
print(f"[OK] {changes} bozuk string satırı düzeltildi.")
