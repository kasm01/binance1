from pathlib import Path

path = Path("main.py")
text = path.read_text()

start = text.find("HybridMultiTFModel(")
end = text.find("whale_detector = MultiTimeframeWhaleDetector", start)

if start == -1 or end == -1:
    print("[ERR] MTF / whale markerları bulunamadı, script uygulanmadı.")
    raise SystemExit(1)

segment = text[start:end]

old = "        system_logger.info("
if old not in segment:
    print("[WARN] İlgili blokta 'system_logger.info(' satırı bulunamadı, dokunulmadı.")
    raise SystemExit(0)

new = segment.replace(
    old,
    "        if system_logger:\n"
    "            system_logger.info(",
    1,
)

text = text[:start] + new + text[end:]
path.write_text(text)
print("[OK] MTF ensemble logger çağrısı korumalı hale getirildi.")
