# tg_bot/formatters.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

def fmt_kv(title: str, rows: List[tuple[str, Any]]) -> str:
    lines = [f"*{title}*"]
    for k, v in rows:
        lines.append(f"• *{k}:* `{v}`")
    return "\n".join(lines)

def emoji_signal(sig: str) -> str:
    s = (sig or "").upper()
    if s == "BUY":
        return "✅"
    if s == "SELL":
        return "🟣"
    if s == "HOLD":
        return "⏸"
    return "❔"

def fmt_status(payload: Dict[str, Any]) -> str:
    """
    payload: main loop'tan doldurabileceğin bir dict.
    şimdilik minimum alanlarla da çalışır.
    """
    symbol = payload.get("symbol", "N/A")
    sig = payload.get("signal", "N/A")
    ens_p = payload.get("ensemble_p", "N/A")
    itvs = payload.get("intervals", [])
    aucs = payload.get("aucs", {})  # {itv: auc}
    parts = []
    parts.append(f"{emoji_signal(sig)} *{symbol}*  |  *Signal:* `{sig}`  |  *Ensemble:* `{ens_p}`")
    if itvs:
        parts.append(f"⏱ *MTF:* `{','.join(itvs)}`")
    if isinstance(aucs, dict) and aucs:
        # kısa göster
        short = ", ".join([f"{k}:{aucs[k]:.3f}" if isinstance(aucs[k], (int,float)) else f\"{k}:{aucs[k]}\" for k in list(aucs.keys())[:6]])
        parts.append(f"📈 *AUC:* `{short}`")
    why = payload.get("why")
    if why:
        parts.append(f"🧩 *Why:* `{why}`")
    return "\n".join(parts)
