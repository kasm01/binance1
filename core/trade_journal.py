from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


class TradeJournal:
    """
    CSV journal writer.
    Amaç: meta içindeki alanları tek bir 'meta' kolonu yerine, kolonlara açarak yazmak (flatten).
    Böylece bt_p_buy_raw / bt_p_buy_ema / bt_ema_alpha gibi alanlar gerçekten CSV header'a girer.
    """

    def __init__(self, path: str = "logs/trade_journal.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Header'ı ilk yazışta dinamik oluşturacağız.
        # Eğer dosya yoksa header yazar; varsa mevcut header ile append eder.
        self._header: Optional[list[str]] = None
        if self.path.exists():
            try:
                with self.path.open("r", newline="", encoding="utf-8") as f:
                    r = csv.reader(f)
                    self._header = next(r, None)
            except Exception:
                self._header = None

    def _flatten_meta(self, meta: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not isinstance(meta, dict):
            return out

        # 1) meta'nın düz anahtarlarını yaz
        for k, v in meta.items():
            if k in ("probs", "extra"):
                continue
            out[k] = v

        # 2) probs içinden seçili alanları kolonlaştır
        probs = meta.get("probs")
        if isinstance(probs, dict):
            # “bt_” prefix ile standardize edelim
            if probs.get("p_buy_raw") is not None:
                out["bt_p_buy_raw"] = float(probs.get("p_buy_raw"))
            else:
                out.setdefault("bt_p_buy_raw", None)

            if probs.get("p_buy_ema") is not None:
                out["bt_p_buy_ema"] = float(probs.get("p_buy_ema"))
            else:
                out.setdefault("bt_p_buy_ema", None)

            # mevcutlarda da işine yarar:
            if probs.get("p_used") is not None:
                out["bt_p_used"] = float(probs.get("p_used"))
            if probs.get("p_single") is not None:
                out["bt_p_single"] = float(probs.get("p_single"))

        # 3) extra içinden seçili alanları kolonlaştır
        extra = meta.get("extra")
        if isinstance(extra, dict):
            if extra.get("ema_alpha") is not None:
                out["bt_ema_alpha"] = float(extra.get("ema_alpha"))
            else:
                out.setdefault("bt_ema_alpha", None)

            # debug büyükse csv şişmesin: json olarak tek kolonda sakla (isteğe bağlı)
            if "mtf_debug" in extra:
                try:
                    out["bt_mtf_debug_json"] = json.dumps(extra.get("mtf_debug"), ensure_ascii=False)
                except Exception:
                    out["bt_mtf_debug_json"] = None

        # Kolonların kesin var olması için (header tutarlılığı):
        out.setdefault("bt_p_buy_raw", None)
        out.setdefault("bt_p_buy_ema", None)
        out.setdefault("bt_ema_alpha", None)

        return out

    def append(self, row: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        # base row
        out: Dict[str, Any] = dict(row)

        # meta flatten
        if meta is None:
            meta = out.get("meta") if isinstance(out.get("meta"), dict) else None
        flat = self._flatten_meta(meta)
        out.update(flat)

        # "meta" kolonunu istemiyorsan kaldır (opsiyonel)
        # out.pop("meta", None)

        # header oluştur/ genişlet
        keys = list(out.keys())
        if self._header is None:
            self._header = keys
            write_header = True
        else:
            write_header = False
            # yeni kolon geldiyse header'ı genişlet (mevcut dosyada header eskiyse sorun çıkarır)
            # Bu yüzden biz dosyayı baştan yedekleyip sıfırlamayı önerdik.
            new_cols = [k for k in keys if k not in self._header]
            if new_cols:
                self._header.extend(new_cols)

        # yaz
        mode = "a" if self.path.exists() else "w"
        with self.path.open(mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._header, extrasaction="ignore")
            if write_header and mode == "w":
                w.writeheader()
            w.writerow(out)
