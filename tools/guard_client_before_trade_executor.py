from pathlib import Path

path = Path("main.py")
text = path.read_text()
lines = text.splitlines(keepends=True)

target = "trade_executor = TradeExecutor("
inserted = False
new_lines = []

for i, line in enumerate(lines):
    if target in line and not inserted:
        indent = line[: len(line) - len(line.lstrip())]
        guard_block = [
            f"{indent}try:\n",
            f"{indent}    client\n",
            f"{indent}except NameError:\n",
            f"{indent}    client = None\n",
        ]
        new_lines.extend(guard_block)
        inserted = True
    new_lines.append(line)

if not inserted:
    print("[WARN] 'trade_executor = TradeExecutor(' satırı bulunamadı, değişiklik yapılmadı.")
else:
    path.write_text("".join(new_lines))
    print("[OK] client guard bloğu trade_executor öncesine eklendi.")
