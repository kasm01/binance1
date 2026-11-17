"""Strategy-to-coin allocation for isolated 8-engine deployment.

Only explicitly assigned pairs are executable. No L2-4/S2-4 defaults.
"""
from __future__ import annotations
ASSIGNED_SYMBOLS={"LONG1":"NEARUSDT","SHORT1":"INJUSDT"}
def validate_assignment(strategy_id:str,symbol:str)->tuple[bool,str]:
    strategy=str(strategy_id or "").strip().upper()
    ticker=str(symbol or "").strip().upper()
    if strategy not in ASSIGNED_SYMBOLS:
        return False,"strategy_symbol_not_assigned"
    expected=ASSIGNED_SYMBOLS[strategy]
    if ticker!=expected:
        return False,"strategy_symbol_mismatch"
    if len(ASSIGNED_SYMBOLS.values())!=len(set(ASSIGNED_SYMBOLS.values())):
        return False,"duplicate_strategy_coin"
    return True,"assigned_strategy_coin"
def validate_8_assignments(plan:dict[str,str])->tuple[bool,str]:
    required={*(f"LONG{i}" for i in range(1,5)),*(f"SHORT{i}" for i in range(1,5))}
    normalized={str(k).upper().strip():str(v).upper().strip() for k,v in plan.items()}
    if set(normalized)!=required:return False,"missing_or_unknown_strategies"
    if any(not v.endswith("USDT") for v in normalized.values()):return False,"unexpected_symbol"
    if len(set(normalized.values()))!=8:return False,"duplicate_symbol"
    for k,v in ASSIGNED_SYMBOLS.items():
        if normalized[k]!=v:return False,"existing_strategy_coin_changed"
    return True,"unique_coin_allocation"
