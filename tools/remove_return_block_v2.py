from pathlib import Path

path = Path("main.py")
text = path.read_text()

old_block = '''    return {
        "symbol": symbol,
        "interval": interval,
        "client": client,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
        "hybrid_model": hybrid_model,
        "mtf_model": mtf_model,
        "whale_detector": whale_detector,
    }

'''

if old_block not in text:
    print("Uyarı: return bloğu tam olarak bu içerikle bulunamadı, dosya değişmedi.")
else:
    backup = path.with_suffix(".py.bak_return_block")
    backup.write_text(text)
    new_text = text.replace(old_block, "")
    path.write_text(new_text)
    print(f"return bloğu kaldırıldı. Yedek: {backup}")
