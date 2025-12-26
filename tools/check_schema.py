#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

META_DIR = Path("models")
INTERVALS = ["1m", "5m", "15m", "1h"]


def load_schema(interval: str) -> List[str]:
    meta_path = META_DIR / f"model_meta_{interval}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta bulunamadı: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    schema = meta.get("feature_schema")
    if not isinstance(schema, list):
        raise ValueError(f"{meta_path} içinde feature_schema yok veya liste değil")
    return schema


def main() -> None:
    schemas: Dict[str, List[str]] = {}
    schema_sets: Dict[str, Set[str]] = {}

    print("🔍 Feature schema kontrolü başlıyor...\n")

    # yükle
    for tf in INTERVALS:
        schema = load_schema(tf)
        schemas[tf] = schema
        schema_sets[tf] = set(schema)
        print(f"✔ {tf}: {len(schema)} feature")

    print("\n" + "-" * 60)

    # referans = 5m (orta TF en mantıklısı)
    ref_tf = "5m"
    ref_schema = schemas[ref_tf]
    ref_set = schema_sets[ref_tf]

    ok = True

    for tf in INTERVALS:
        if tf == ref_tf:
            continue

        s = schema_sets[tf]

        missing = sorted(ref_set - s)
        extra = sorted(s - ref_set)

        if not missing and not extra:
            print(f"✅ {tf} == {ref_tf}  (schema aynı)")
        else:
            ok = False
            print(f"❌ {tf} != {ref_tf}")
            if missing:
                print(f"   - eksik ({len(missing)}): {missing}")
            if extra:
                print(f"   - fazla ({len(extra)}): {extra}")

    print("\n" + "-" * 60)

    # sıralama farkı kontrolü (çok önemli!)
    for tf in INTERVALS:
        if schemas[tf] != ref_schema:
            print(f"⚠️  {tf} sıralama farkı var (order mismatch)")
            ok = False

    print("\n" + "=" * 60)
    if ok:
        print("🎉 RESULT: OK — Tüm feature_schema'lar birebir uyumlu")
    else:
        print("🚨 RESULT: FAIL — MTF ensemble AÇILMAMALI")
        print("👉 Eğitim / pipeline schema'ları eşitlemeden USE_MTF_ENS=true yapma")


if __name__ == "__main__":
    main()
