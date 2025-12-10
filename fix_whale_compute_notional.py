import re
from pathlib import Path

path = Path("core/trade_executor.py")
text = path.read_text()

new_block = """    def _compute_notional(
        self,
        symbol: str,
        signal: str,
        price: float,
        extra: Dict[str, Any],
    ) -> float:
        \"""
        Basit notional hesaplama:
          - base_order_notional
          - whale / model_conf vs. ile çarpılabilir
          - max_position_notional'a clamp edilir
        \"""

        notional = self.base_order_notional

        # Model güven faktörü
        model_conf = float(extra.get("model_confidence_factor", 1.0) or 1.0)
        notional *= model_conf

        # Whale bilgisi (varsayılan: yok)
        # main.py içinde extra["whale_meta"] şöyle dolduruluyor:
        # {
        #   "direction": "long" / "short" / "none",
        #   "score": float,
        #   ...
        # }
        # Eski kodla geri uyumluluk için extra["whale"] da fallback olarak okunuyor.
        whale_info = extra.get("whale_meta") or extra.get("whale") or {}
        whale_score = float(whale_info.get("score", 0.0) or 0.0)
        whale_direction = whale_info.get("direction")  # "long" / "short" / "none"

        # Eğer whale_score yüksekse ve sinyal ile aynı yöndeyse notional'ı boost et
        if whale_score > 0 and whale_direction in ("long", "short"):
            if signal == whale_direction:
                notional *= (1.0 + self.whale_risk_boost * whale_score)

        # max_position_notional sınırı
"""

pattern = r"    def _compute_notional\([^\n]*\n(?:.*\n)*?        # max_position_notional sınırı\n"
new_text, n = re.subn(pattern, new_block, text, count=1, flags=re.DOTALL)
if n == 0:
    raise SystemExit("Pattern not found; _compute_notional signature may have changed.")
path.write_text(new_text)
print("Patched _compute_notional successfully.")
