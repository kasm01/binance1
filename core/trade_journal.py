import os
import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None
    Credentials = None


class TradeJournal:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("system")

        self.csv_path = os.getenv("TRADE_JOURNAL_CSV_PATH", "logs/trade_journal.csv")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        # Google Sheet opsiyonel
        self.gs_enabled = os.getenv("ENABLE_GSHEET_JOURNAL", "0") == "1"
        self.gs_credentials_path = os.getenv("GSHEET_CREDENTIALS_PATH", "")
        self.gs_spreadsheet_id = os.getenv("GSHEET_TRADES_SHEET_ID", "")
        self.gs_worksheet_name = os.getenv("GSHEET_TRADES_WORKSHEET", "Trades")

        self.gs_client = None
        self.gs_sheet = None

        if self.gs_enabled:
            self._init_gsheet()

        # CSV için header varsa kontrol
        self._ensure_csv_header()

    # --------------------------------------------------------------
    # Google Sheet
    # --------------------------------------------------------------
    def _init_gsheet(self) -> None:
        if not self.gs_credentials_path or not self.gs_spreadsheet_id:
            self.logger.warning(
                "[JOURNAL] ENABLE_GSHEET_JOURNAL=1 ama credentials veya sheet id yok."
            )
            self.gs_enabled = False
            return
        if gspread is None or Credentials is None:
            self.logger.warning(
                "[JOURNAL] gspread / google-auth yok. `pip install gspread google-auth`."
            )
            self.gs_enabled = False
            return
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_file(
                self.gs_credentials_path, scopes=scopes
            )
            self.gs_client = gspread.authorize(creds)
            ss = self.gs_client.open_by_key(self.gs_spreadsheet_id)
            self.gs_sheet = ss.worksheet(self.gs_worksheet_name)
            self.logger.info("[JOURNAL] Google Sheet bağlandı.")
        except Exception as e:
            self.logger.warning("[JOURNAL] Google Sheet bağlantı hatası: %s", e)
            self.gs_enabled = False

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------
    def _ensure_csv_header(self) -> None:
        try:
            # dosya yoksa header yaz
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "timestamp",
                            "symbol",
                            "side",
                            "interval",
                            "entry_price",
                            "exit_price",
                            "qty",
                            "notional",
                            "pnl_usdt",
                            "pnl_pct",
                            "duration_min",
                            "reason",
                            "label_best_side",
                            "model_conf",
                        ]
                    )
        except Exception as e:
            self.logger.warning("[JOURNAL] CSV header yazılamadı: %s", e)

    def _append_csv_row(self, row: Dict[str, Any]) -> None:
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        row.get("timestamp"),
                        row.get("symbol"),
                        row.get("side"),
                        row.get("interval"),
                        row.get("entry_price"),
                        row.get("exit_price"),
                        row.get("qty"),
                        row.get("notional"),
                        row.get("pnl_usdt"),
                        row.get("pnl_pct"),
                        row.get("duration_min"),
                        row.get("reason"),
                        row.get("label_best_side"),
                        row.get("model_conf"),
                    ]
                )
        except Exception as e:
            self.logger.warning("[JOURNAL] CSV append hata: %s", e)

    def _append_gsheet_row(self, row: Dict[str, Any]) -> None:
        if not self.gs_enabled or self.gs_sheet is None:
            return
        try:
            self.gs_sheet.append_row(
                [
                    row.get("timestamp"),
                    row.get("symbol"),
                    row.get("side"),
                    row.get("interval"),
                    row.get("entry_price"),
                    row.get("exit_price"),
                    row.get("qty"),
                    row.get("notional"),
                    row.get("pnl_usdt"),
                    row.get("pnl_pct"),
                    row.get("duration_min"),
                    row.get("reason"),
                    row.get("label_best_side"),
                    row.get("model_conf"),
                ],
                value_input_option="RAW",
            )
        except Exception as e:
            self.logger.warning("[JOURNAL] GSheet append hata: %s", e)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def log_trade_from_close(
        self,
        *,
        symbol: str,
        exit_price: float,
        pnl_usdt: float,
        meta: Dict[str, Any],
    ) -> None:
        """
        RiskManager.on_position_close'tan çağrılacak.
        meta içinde:
        - reason
        - closed_side
        - opened_at
        - interval
        - qty
        - entry_price
        - notional
        - probs_at_close
        - extra
        """
        try:
            side = meta.get("closed_side")
            opened_at_str = meta.get("opened_at")
            interval = meta.get("interval")
            qty = float(meta.get("qty", 0.0) or 0.0)
            entry_price = float(meta.get("entry_price", 0.0) or 0.0)
            notional = float(meta.get("notional", qty * entry_price) or 0.0)
            reason = meta.get("reason", "")
            probs = meta.get("probs_at_close") or {}
            best_side = meta.get("extra", {}).get("best_side")
            model_conf = float(meta.get("extra", {}).get("model_confidence_factor", 1.0))

            ts_close = datetime.utcnow()
            if opened_at_str:
                try:
                    ts_open = datetime.fromisoformat(opened_at_str)
                except Exception:
                    ts_open = ts_close
            else:
                ts_open = ts_close
            duration_min = (ts_close - ts_open).total_seconds() / 60.0

            pnl_pct = 0.0
            if notional > 0:
                pnl_pct = pnl_usdt / notional

            row = {
                "timestamp": ts_close.isoformat(),
                "symbol": symbol,
                "side": side,
                "interval": interval,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "notional": notional,
                "pnl_usdt": pnl_usdt,
                "pnl_pct": pnl_pct,
                "duration_min": duration_min,
                "reason": reason,
                "label_best_side": best_side,
                "model_conf": model_conf,
            }

            self._append_csv_row(row)
            self._append_gsheet_row(row)
            self.logger.info(
                "[JOURNAL] Trade loglandı | symbol=%s side=%s pnl=%.4f notional=%.2f",
                symbol,
                side,
                pnl_usdt,
                notional,
            )
        except Exception as e:
            self.logger.warning("[JOURNAL] log_trade_from_close hata: %s", e)
