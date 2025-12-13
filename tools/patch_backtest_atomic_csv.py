import re
from pathlib import Path

BT = Path("backtest_mtf.py")
txt = BT.read_text(encoding="utf-8")

# 1) atomic helper ekle (yoksa)
if "_atomic_to_csv" not in txt:
    insert_after = None

    # import blokunun sonunu yakalamaya çalış
    m = re.search(r"(from\s+dataclasses\s+import\s+dataclass\s*\n)", txt)
    if m:
        insert_after = m.end()
    else:
        # fallback: en başa yakın bir yere koy
        m2 = re.search(r"(import\s+pandas\s+as\s+pd\s*\n)", txt)
        insert_after = m2.end() if m2 else 0

    helper = r'''

# ==========================================================
# Atomic CSV writer (boş/yarım dosya kalmasın)
# ==========================================================
def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # df boşsa header bile yazmıyordu -> 1 byte dosya üretiyordu.
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    # bazı kolonlar yoksa ekle
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]
'''
    txt = txt[:insert_after] + helper + txt[insert_after:]

# 2) Export CSV bloğunu atomic + header garantili hale getir
# - mevcut 3 to_csv satırını yakalayıp değiştiriyoruz
pattern = re.compile(
    r"""
(?P<indent>^[ \t]*)pd\.DataFrame\(equity_rows\)\.to_csv\(\s*equity_path\s*,\s*index=False\s*\)\s*\n
(?P=indent)pd\.DataFrame\(closed_trades\)\.to_csv\(\s*trades_path\s*,\s*index=False\s*\)\s*\n
(?P=indent)pd\.DataFrame\(\[bt_stats\.summary_dict\(\)\]\)\.to_csv\(\s*summary_path\s*,\s*index=False\s*\)\s*
""",
    re.VERBOSE | re.MULTILINE,
)

m = pattern.search(txt)
if not m:
    raise SystemExit("[ERR] Export CSV bloğu bulunamadı. backtest_mtf.py içindeki 3 to_csv satırı farklı görünüyor.")

indent = m.group("indent")

trades_cols = [
    "symbol","side","qty","entry_price","notional","interval","opened_at",
    "sl_price","tp_price","trailing_pct","atr_value","highest_price","lowest_price",
    "meta","closed_at","close_price","realized_pnl","close_reason",
    "bt_symbol","bt_interval","bt_bar","bt_time","bt_signal","bt_price",
    "bt_p_used","bt_p_single","bt_p_1m","bt_p_5m","bt_p_15m","bt_p_1h",
    "whale_dir","whale_score","bt_atr",
]

equity_cols = [
    "bar","time","symbol","interval","price","signal",
    "p_used","p_single","p_1m","p_5m","p_15m","p_1h",
    "whale_dir","whale_score","atr",
    "equity","peak_equity","max_drawdown_pct","pnl_total",
]

replacement = (
    f"{indent}eq_df = _ensure_columns(pd.DataFrame(equity_rows), {equity_cols!r})\n"
    f"{indent}tr_df = _ensure_columns(pd.DataFrame(closed_trades), {trades_cols!r})\n"
    f"{indent}sm_df = pd.DataFrame([bt_stats.summary_dict()])\n"
    f"{indent}_atomic_to_csv(eq_df, equity_path)\n"
    f"{indent}_atomic_to_csv(tr_df, trades_path)\n"
    f"{indent}_atomic_to_csv(sm_df, summary_path)\n"
)

txt = txt[:m.start()] + replacement + txt[m.end():]
BT.write_text(txt, encoding="utf-8")

print("[OK] backtest_mtf.py: atomic csv + empty header fix eklendi.")
