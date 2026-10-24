from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: Any, default: float = 1.0) -> float:
    return max(0.0, min(1.0, _f(value, default)))


@dataclass(frozen=True)
class CentralRiskConfig:
    """Shared portfolio-risk policy.

    Exposure caps are optional only for research/shadow. Production must
    provide explicit tested caps and pass validate_for_production().
    """
    dd_reduce_trigger: float = 0.08
    dd_recover_trigger: float = 0.04
    normal_multiplier: float = 1.00
    recovery_multiplier: float = 0.70
    reduced_multiplier: float = 0.50
    recovery_successes_required: int = 2
    # Portfolio-level operational sizing buffer applied to every new entry
    # before hard exposure caps. Keep default 1.0; validated shadow profiles
    # may reserve headroom for mark-to-market notional drift.
    operational_headroom_multiplier: float = 1.00
    max_open_positions: int = 8
    min_order_notional: float = 10.0
    max_gross_exposure_ratio: Optional[float] = None
    max_long_exposure_ratio: Optional[float] = None
    max_short_exposure_ratio: Optional[float] = None
    max_abs_net_exposure_ratio: Optional[float] = None
    max_symbol_exposure_ratio: Optional[float] = None
    # Early telemetry warning before a hard exposure cap is reached. This does
    # not resize/close positions; it only makes continuous exposure health
    # explicit for the central controller.
    exposure_warning_fraction: float = 0.95
    # Explicit human/release authorization. Research/shadow profiles must keep
    # this False even when all numeric caps are populated.
    production_authorized: bool = False

    def validate_for_production(self) -> Tuple[bool, Tuple[str, ...]]:
        issues: List[str] = []
        if not bool(self.production_authorized):
            issues.append("production_not_authorized")
        for name in (
            "max_gross_exposure_ratio",
            "max_long_exposure_ratio",
            "max_short_exposure_ratio",
            "max_abs_net_exposure_ratio",
            "max_symbol_exposure_ratio",
        ):
            if getattr(self, name) is None:
                issues.append(name)
        if self.dd_reduce_trigger <= self.dd_recover_trigger:
            issues.append("dd_threshold_order")
        if self.recovery_successes_required < 1:
            issues.append("recovery_successes_required")
        if not (0.0 < float(self.operational_headroom_multiplier) <= 1.0):
            issues.append("operational_headroom_multiplier")
        if not (0.0 < float(self.exposure_warning_fraction) <= 1.0):
            issues.append("exposure_warning_fraction")
        return (not issues), tuple(issues)


@dataclass
class PortfolioRiskState:
    equity_peak: float
    mode: str = "NORMAL"
    recovery_successes: int = 0

    def normalized_mode(self) -> str:
        mode = str(self.mode or "NORMAL").strip().upper()
        return mode if mode in {"NORMAL", "RECOVERY", "REDUCED"} else "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity_peak": float(self.equity_peak),
            "mode": self.normalized_mode(),
            "recovery_successes": int(self.recovery_successes),
        }


@dataclass(frozen=True)
class PositionExposure:
    strategy_id: str
    symbol: str
    side: str
    notional: float
    opened_risk_mode: str = "NORMAL"

    def side_norm(self) -> str:
        return str(self.side or "").strip().lower()


@dataclass(frozen=True)
class EntryRiskRequest:
    strategy_id: str
    symbol: str
    side: str
    requested_notional: float
    market_context_multiplier: float = 1.0


@dataclass(frozen=True)
class ExposureSnapshot:
    gross_notional: float
    long_notional: float
    short_notional: float
    net_notional: float
    symbol_notional: float
    open_positions: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureHealth:
    schema_version: str
    status: str
    portfolio_equity: float
    gross_ratio: float
    long_ratio: float
    short_ratio: float
    abs_net_ratio: float
    breached_caps: Tuple[str, ...]
    warning_caps: Tuple[str, ...]
    freeze_new_entries: bool

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["breached_caps"] = list(self.breached_caps)
        out["warning_caps"] = list(self.warning_caps)
        return out


@dataclass(frozen=True)
class CentralRiskDecision:
    schema_version: str
    state: str
    strategy_id: str
    symbol: str
    side: str
    portfolio_mode: str
    state_multiplier: float
    market_context_multiplier: float
    operational_headroom_multiplier: float
    requested_notional: float
    risk_scaled_notional: float
    allowed_notional: float
    reason_codes: Tuple[str, ...]
    exposure_before: ExposureSnapshot
    cap_headroom: Dict[str, Optional[float]]
    production_ready: bool

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["reason_codes"] = list(self.reason_codes)
        out["exposure_before"] = self.exposure_before.to_dict()
        return out


class CentralRiskMaster:
    """Pure shared risk core for LONG1..4 + SHORT1..4.

    It places no orders and owns no exchange/Redis I/O. Runtime state and
    persistence can wrap this deterministic contract later.
    """

    def __init__(self, config: Optional[CentralRiskConfig] = None) -> None:
        self.config = config or CentralRiskConfig()

    def state_multiplier(self, mode: str) -> float:
        mode = str(mode or "NORMAL").strip().upper()
        if mode == "REDUCED":
            return _clamp01(self.config.reduced_multiplier, 0.50)
        if mode == "RECOVERY":
            return _clamp01(self.config.recovery_multiplier, 0.70)
        return _clamp01(self.config.normal_multiplier, 1.00)

    def update_state_on_mtm(
        self,
        *,
        state: PortfolioRiskState,
        mtm_equity: float,
    ) -> Tuple[PortfolioRiskState, Tuple[str, ...]]:
        """Update only portfolio DD regime from mark-to-market equity.

        Recovery-success trade counting remains the responsibility of
        update_state_on_close(); this method is safe to call every telemetry tick.
        """
        cfg = self.config
        mtm = max(0.0, _f(mtm_equity))
        peak = max(_f(state.equity_peak, mtm), mtm)
        mode = state.normalized_mode()
        wins = max(0, int(state.recovery_successes or 0))
        dd = (mtm / peak - 1.0) if peak > 0.0 else 0.0
        reasons: List[str] = []

        if mode == "NORMAL" and dd <= -abs(float(cfg.dd_reduce_trigger)):
            mode, wins = "REDUCED", 0
            reasons.append("mtm_dd_enter_reduced")
        elif mode == "REDUCED" and dd >= -abs(float(cfg.dd_recover_trigger)):
            mode, wins = "RECOVERY", 0
            reasons.append("mtm_dd_enter_recovery")
        elif mode == "RECOVERY" and dd <= -abs(float(cfg.dd_reduce_trigger)):
            mode, wins = "REDUCED", 0
            reasons.append("mtm_recovery_relapse")

        if not reasons:
            reasons.append("state_unchanged")
        return PortfolioRiskState(float(peak), mode, int(wins)), tuple(reasons)

    def update_state_on_close(
        self,
        *,
        state: PortfolioRiskState,
        equity_after: float,
        pnl_net: float,
        opened_risk_mode: str,
        is_risk_stop: bool = False,
    ) -> Tuple[PortfolioRiskState, Tuple[str, ...]]:
        cfg = self.config
        equity = max(0.0, _f(equity_after))
        peak = max(_f(state.equity_peak, equity), equity)
        mode = state.normalized_mode()
        wins = max(0, int(state.recovery_successes or 0))
        reasons: List[str] = []
        dd = (equity / peak - 1.0) if peak > 0.0 else 0.0

        if mode == "NORMAL":
            if dd <= -abs(float(cfg.dd_reduce_trigger)):
                mode, wins = "REDUCED", 0
                reasons.append("dd_enter_reduced")
        elif mode == "REDUCED":
            if dd >= -abs(float(cfg.dd_recover_trigger)):
                mode, wins = "RECOVERY", 0
                reasons.append("dd_enter_recovery")
        elif mode == "RECOVERY":
            if is_risk_stop or dd <= -abs(float(cfg.dd_reduce_trigger)):
                mode, wins = "REDUCED", 0
                reasons.append("recovery_relapse")
            elif (
                str(opened_risk_mode or "").strip().upper() == "RECOVERY"
                and _f(pnl_net) > 0.0
            ):
                wins += 1
                reasons.append("recovery_success")
                if wins >= max(1, int(cfg.recovery_successes_required)):
                    mode, wins = "NORMAL", 0
                    reasons.append("recovery_confirmed_normal")

        if not reasons:
            reasons.append("state_unchanged")
        return PortfolioRiskState(float(peak), mode, int(wins)), tuple(reasons)

    @staticmethod
    def exposure_snapshot(
        positions: Iterable[PositionExposure], *, symbol: str
    ) -> ExposureSnapshot:
        gross = long_n = short_n = symbol_n = 0.0
        count = 0
        symbol = str(symbol or "").upper().strip()
        for pos in positions:
            notional = max(0.0, _f(pos.notional))
            side = pos.side_norm()
            if notional <= 0.0 or side not in {"long", "short"}:
                continue
            count += 1
            gross += notional
            if side == "long":
                long_n += notional
            else:
                short_n += notional
            if str(pos.symbol or "").upper().strip() == symbol:
                symbol_n += notional
        return ExposureSnapshot(
            gross_notional=float(gross),
            long_notional=float(long_n),
            short_notional=float(short_n),
            net_notional=float(long_n - short_n),
            symbol_notional=float(symbol_n),
            open_positions=int(count),
        )

    def evaluate_exposure_health(
        self,
        *,
        portfolio_equity: float,
        positions: Iterable[PositionExposure],
    ) -> ExposureHealth:
        """Continuously classify marked exposure against portfolio caps.

        Callers should pass mark-to-market position notionals (qty * mark), not
        entry notionals. A BREACH freezes risk-increasing new entries; this
        method deliberately does not force-close positions by itself.
        """
        cfg = self.config
        equity = max(0.0, _f(portfolio_equity))
        snap = self.exposure_snapshot(list(positions), symbol="")
        denom = max(equity, 1e-12)
        ratios = {
            "gross": snap.gross_notional / denom,
            "long": snap.long_notional / denom,
            "short": snap.short_notional / denom,
            "abs_net": abs(snap.net_notional) / denom,
        }
        caps = {
            "gross": cfg.max_gross_exposure_ratio,
            "long": cfg.max_long_exposure_ratio,
            "short": cfg.max_short_exposure_ratio,
            "abs_net": cfg.max_abs_net_exposure_ratio,
        }
        warn_frac = max(0.0, min(1.0, _f(cfg.exposure_warning_fraction, 0.95)))
        breached: List[str] = []
        warning: List[str] = []
        if equity <= 0.0:
            breached.append("invalid_equity")
        for name, cap in caps.items():
            if cap is None:
                continue
            cap_f = max(0.0, float(cap))
            value = float(ratios[name])
            if value > cap_f + 1e-12:
                breached.append(name)
            elif cap_f > 0.0 and value >= cap_f * warn_frac:
                warning.append(name)
        status = "BREACH" if breached else ("WARNING" if warning else "OK")
        return ExposureHealth(
            schema_version="central_exposure_health.v1",
            status=status,
            portfolio_equity=float(equity),
            gross_ratio=float(ratios["gross"]),
            long_ratio=float(ratios["long"]),
            short_ratio=float(ratios["short"]),
            abs_net_ratio=float(ratios["abs_net"]),
            breached_caps=tuple(breached),
            warning_caps=tuple(warning),
            freeze_new_entries=bool(breached),
        )

    @staticmethod
    def _cap_headroom(
        cap_ratio: Optional[float], equity: float, used: float
    ) -> Optional[float]:
        if cap_ratio is None:
            return None
        return max(0.0, float(cap_ratio) * equity - max(0.0, used))

    @staticmethod
    def _net_headroom(
        cap_ratio: Optional[float], equity: float, net: float, side: str
    ) -> Optional[float]:
        if cap_ratio is None:
            return None
        cap = max(0.0, float(cap_ratio) * equity)
        sign = 1.0 if side == "long" else -1.0
        aligned = sign * net >= 0.0
        return max(0.0, cap - abs(net)) if aligned else max(0.0, cap + abs(net))


    def evaluate_entry(
        self,
        *,
        request: EntryRiskRequest,
        state: PortfolioRiskState,
        portfolio_equity: float,
        positions: Iterable[PositionExposure],
    ) -> CentralRiskDecision:
        cfg = self.config
        equity = max(0.0, _f(portfolio_equity))
        side = str(request.side or "").strip().lower()
        strategy_id = str(request.strategy_id or "").strip().upper()
        symbol = str(request.symbol or "").strip().upper()
        positions = list(positions)
        exposure = self.exposure_snapshot(positions, symbol=symbol)
        production_ready, _ = cfg.validate_for_production()
        reasons: List[str] = []

        mode = state.normalized_mode()
        state_mult = self.state_multiplier(mode)
        context_mult = _clamp01(request.market_context_multiplier, 1.0)
        headroom_mult = _clamp01(cfg.operational_headroom_multiplier, 1.0)
        requested = max(0.0, _f(request.requested_notional))
        scaled = requested * headroom_mult * state_mult * context_mult

        if not strategy_id or not symbol or side not in {"long", "short"}:
            reasons.append("invalid_request")
        if equity <= 0.0:
            reasons.append("invalid_portfolio_equity")
        if requested <= 0.0:
            reasons.append("invalid_requested_notional")
        if any(
            str(pos.strategy_id or "").strip().upper() == strategy_id
            for pos in positions
        ):
            reasons.append("strategy_position_exists")
        if exposure.open_positions >= max(1, int(cfg.max_open_positions)):
            reasons.append("max_open_positions")

        side_used = (
            exposure.long_notional if side == "long" else exposure.short_notional
        )
        cap_headroom: Dict[str, Optional[float]] = {
            "gross": self._cap_headroom(
                cfg.max_gross_exposure_ratio, equity, exposure.gross_notional
            ),
            "side": self._cap_headroom(
                cfg.max_long_exposure_ratio
                if side == "long"
                else cfg.max_short_exposure_ratio,
                equity,
                side_used,
            ),
            "net": self._net_headroom(
                cfg.max_abs_net_exposure_ratio,
                equity,
                exposure.net_notional,
                side,
            ),
            "symbol": self._cap_headroom(
                cfg.max_symbol_exposure_ratio, equity, exposure.symbol_notional
            ),
        }

        allowed = scaled
        for cap_name, room in cap_headroom.items():
            if room is not None and allowed > room:
                allowed = room
                reasons.append(f"{cap_name}_cap_limited")

        hard_reject = any(
            code in reasons
            for code in (
                "invalid_request",
                "invalid_portfolio_equity",
                "invalid_requested_notional",
                "strategy_position_exists",
                "max_open_positions",
            )
        )
        if hard_reject:
            allowed = 0.0
        if 0.0 < allowed < float(cfg.min_order_notional):
            allowed = 0.0
            reasons.append("below_min_notional")

        if mode == "REDUCED":
            reasons.append("portfolio_reduced")
        elif mode == "RECOVERY":
            reasons.append("portfolio_recovery")
        if context_mult < 1.0:
            reasons.append("market_context_reduced")
        if headroom_mult < 1.0:
            reasons.append("operational_headroom_reduced")
        if not reasons:
            reasons.append("normal")

        return CentralRiskDecision(
            schema_version="central_risk_decision.v1",
            state="READY" if allowed > 0.0 else "REJECT",
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            portfolio_mode=mode,
            state_multiplier=float(state_mult),
            market_context_multiplier=float(context_mult),
            operational_headroom_multiplier=float(headroom_mult),
            requested_notional=float(requested),
            risk_scaled_notional=float(scaled),
            allowed_notional=float(allowed),
            reason_codes=tuple(dict.fromkeys(reasons)),
            exposure_before=exposure,
            cap_headroom=cap_headroom,
            production_ready=bool(production_ready),
        )
