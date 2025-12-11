from pathlib import Path

path = Path("main.py")
text = path.read_text()

start_marker = "# Whale detector (MTF versiyon)"
end_marker = "# -----------------------------\n    # Trade Executor"

start = text.find(start_marker)
end = text.find(end_marker, start)

if start == -1 or end == -1:
    print("[ERR] Markerlar bulunamadı, manuel müdahale gerek.")
    raise SystemExit(1)

new_block = '''    # ------------------------------
    # Whale detector (MTF versiyon)
    # ------------------------------
    whale_detector = MultiTimeframeWhaleDetector()
    if system_logger:
        system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")

'''

# start satırından itibaren, Trade Executor başlığından hemen önceye kadar olan kısmı değiştiriyoruz
text = text[:start] + new_block + text[end:]

path.write_text(text)
print("[OK] Whale detector bloğu temiz şekilde yeniden yazıldı.")
