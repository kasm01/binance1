from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from orchestration.risk.central_risk_master import CentralRiskConfig, CentralRiskMaster


PROFILE_ID = "central_risk_shadow.v1_12_6"
PROFILE_STAGE = "SHADOW_RESEARCH"
VALIDATED_STRATEGIES = ("LONG1", "SHORT1")
TARGET_STRATEGIES = (
    "LONG1", "LONG2", "LONG3", "LONG4",
    "SHORT1", "SHORT2", "SHORT3", "SHORT4",
)


def build_shadow_config() -> CentralRiskConfig:
    """Two-engine MTM-tested research profile; never production-authorized."""
    return CentralRiskConfig(
        dd_reduce_trigger=0.12,
        dd_recover_trigger=0.06,
        normal_multiplier=1.00,
        recovery_multiplier=0.70,
        reduced_multiplier=0.50,
        recovery_successes_required=2,
        operational_headroom_multiplier=0.96,
        max_open_positions=8,
        min_order_notional=10.0,
        max_gross_exposure_ratio=1.00,
        max_long_exposure_ratio=0.50,
        max_short_exposure_ratio=0.50,
        max_abs_net_exposure_ratio=0.50,
        max_symbol_exposure_ratio=0.50,
        exposure_warning_fraction=0.95,
        production_authorized=False,
    )


def build_shadow_master() -> CentralRiskMaster:
    return CentralRiskMaster(build_shadow_config())


def describe_profile() -> Dict[str, Any]:
    cfg = build_shadow_config()
    production_ready, issues = cfg.validate_for_production()
    return {
        "schema_version": "central_risk_shadow_profile.v1",
        "profile_id": PROFILE_ID,
        "stage": PROFILE_STAGE,
        "validated_strategies": list(VALIDATED_STRATEGIES),
        "target_strategies": list(TARGET_STRATEGIES),
        "config": asdict(cfg),
        "production_ready": bool(production_ready),
        "production_issues": list(issues),
        "authority": {
            "portfolio_mtm_dd": True,
            "gross_exposure": True,
            "side_exposure": True,
            "abs_net_exposure": True,
            "symbol_exposure": True,
            "strategy_single_slot": True,
            "order_routing": False,
        },
        "notes": [
            "12/6 DD thresholds are validated only on LONG1+SHORT1 minute MTM replay.",
            "Operational headroom 0.96 was validated on marked-notional LONG1+SHORT1 replay to reserve room for MTM exposure drift.",
            "Exposure caps are research caps for the current two-engine stage, not final eight-engine caps.",
            "Strategy-specific alpha/context multipliers remain inputs; portfolio DD/exposure authority is central.",
        ],
    }
