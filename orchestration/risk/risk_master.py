from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


def _clamp01(value: Any, default: float = 1.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    return max(0.0, min(1.0, v))


@dataclass(frozen=True)
class RiskDecision:
    """Canonical stateless risk decision shared by LONG/SHORT engines.

    Runtime account/portfolio state is intentionally external to this object.
    A future central Risk Master service can own that state while preserving
    this contract unchanged.
    """

    schema_version: str
    state: str
    base_mode: str
    base_multiplier: float
    market_context_multiplier: float
    final_size_multiplier: float
    reason_codes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["reason_codes"] = list(self.reason_codes)
        return out



def resolve_market_context_multiplier(
    *,
    market_context: Any,
    oi_confirmed_build_enable: bool = False,
    oi_confirmed_build_multiplier: Any = 0.50,
) -> Tuple[float, Tuple[str, ...]]:
    """Map canonical MARKET CONTEXT regimes to central risk policy.

    MARKET CONTEXT classifies the market. Risk Master owns what that
    classification means for position size. Unknown/missing context is neutral.
    """
    ctx = market_context if isinstance(market_context, dict) else {}
    regime = ctx.get("regime") if isinstance(ctx.get("regime"), dict) else {}
    name = str(regime.get("name") or ctx.get("oi_state") or "normal").strip().lower()
    if bool(oi_confirmed_build_enable) and name == "confirmed_build":
        return _clamp01(oi_confirmed_build_multiplier, 0.50), ("oi_confirmed_build",)
    return 1.0, ("market_context_normal",)

def evaluate_position_risk(
    *,
    market_context_multiplier: Any = 1.0,
    base_mode: str = "NORMAL",
    normal_multiplier: Any = 1.0,
    reduced_multiplier: Any = 0.50,
) -> RiskDecision:
    """Combine independent risk controls exactly once.

    The caller supplies the account/portfolio mode. MARKET CONTEXT supplies
    only its own multiplier. This function owns the multiplication rule.
    """
    mode = str(base_mode or "NORMAL").strip().upper()
    if mode not in ("NORMAL", "REDUCED"):
        mode = "NORMAL"

    normal = _clamp01(normal_multiplier, 1.0)
    reduced = _clamp01(reduced_multiplier, 0.50)
    base = reduced if mode == "REDUCED" else normal
    context = _clamp01(market_context_multiplier, 1.0)
    final = _clamp01(base * context, 0.0)

    reasons = []
    if mode == "REDUCED":
        reasons.append("drawdown_reduced")
    if context < 1.0:
        reasons.append("market_context_reduced")
    if not reasons:
        reasons.append("normal")

    return RiskDecision(
        schema_version="risk_decision.v1",
        state="READY" if final > 0.0 else "REJECT",
        base_mode=mode,
        base_multiplier=float(base),
        market_context_multiplier=float(context),
        final_size_multiplier=float(final),
        reason_codes=tuple(reasons),
    )
