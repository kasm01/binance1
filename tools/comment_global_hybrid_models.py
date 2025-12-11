from pathlib import Path

path = Path("main.py")
text = path.read_text()
lines = text.splitlines()

out_lines = []
i = 0
changed = False

def comment_call_block(start_line: str, lines, i: int):
    """
    'foo = HybridModel(' gibi bir satırdan başlayıp,
    kapanan ')' satırına kadar hepsini '#' ile comment eder.
    """
    out = []
    changed_local = False

    # ilk satır
    line = lines[i]
    out.append("# " + line)
    changed_local = True
    i += 1

    # devam eden satırlar: kapanan paranteze kadar
    paren_depth = 0
    # ilk satırda '(' say
    paren_depth += start_line.count("(") - start_line.count(")")

    while i < len(lines):
        line = lines[i]
        out.append("# " + line)
        paren_depth += line.count("(") - line.count(")")
        i += 1
        if paren_depth <= 0:
            break

    return out, i, changed_local

while i < len(lines):
    line = lines[i]

    # Sadece GLOBAL scope'taki satırları hedefliyoruz (indent yok)
    stripped = line.lstrip()
    indent_len = len(line) - len(stripped)

    # hybrid_model = HybridModel(...)
    if indent_len == 0 and stripped.startswith("hybrid_model = HybridModel("):
        block, i, changed_local = comment_call_block(line, lines, i)
        out_lines.extend(block)
        changed = changed or changed_local
        continue

    # mtf_model = HybridMultiTFModel(...)
    if indent_len == 0 and stripped.startswith("mtf_model = HybridMultiTFModel("):
        block, i, changed_local = comment_call_block(line, lines, i)
        out_lines.extend(block)
        changed = changed or changed_local
        continue

    # normal satır
    out_lines.append(line)
    i += 1

if not changed:
    print("[INFO] Global scope'ta hybrid_model / mtf_model init bulunamadı, değişiklik yapılmadı.")
else:
    path.write_text("\n".join(out_lines) + "\n")
    print("[OK] Global hybrid_model / mtf_model init blokları comment edildi.")
