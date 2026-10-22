#!/usr/bin/env python3
from __future__ import annotations

import json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from orchestration.event_bus.redis_bus import RedisBus

FAPI = "https://fapi.binance.com"
COLS = ["open_time","open","high","low","close","volume","close_time","quote_asset_volume","number_of_trades","taker_buy_base_volume","taker_buy_quote_volume","ignore"]

def _ef(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)) or default)
    except Exception: return float(default)

def _ei(name: str, default: int) -> int:
    try: return int(float(os.getenv(name, str(default)) or default))
    except Exception: return int(default)

def _fetch(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: List[list] = []; cur = int(start_ms)
    while cur <= end_ms:
        r = session.get(f"{FAPI}/fapi/v1/klines", params={"symbol":symbol,"interval":"1m","startTime":cur,"endTime":int(end_ms),"limit":1500}, timeout=10)
        r.raise_for_status(); batch = r.json()
        if not isinstance(batch, list) or not batch: break
        rows.extend(batch); last = int(batch[-1][0]); cur = last + 60_000
        if len(batch) < 1500: break
        time.sleep(0.05)
    if not rows: return pd.DataFrame(columns=COLS)
    return pd.DataFrame(rows, columns=COLS).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def _sim(d: pd.DataFrame, si: int, entry: float, lev: float, tp: float, gb: float, hard: float, fee: float, horizon: int) -> Tuple[float,str,int,float]:
    best = 0.0; armed = False; hard_px = entry * (1.0 - hard / lev); end = min(len(d), si + horizon)
    for i in range(si, end):
        hi, lo = float(d.iloc[i].high), float(d.iloc[i].low)
        if hi >= hard_px:
            return ((entry-hard_px)/entry)-fee, "hard_stop", i-si+1, best
        if armed:
            trigger = best - gb; trigger_px = entry * (1.0 - trigger / lev)
            if hi >= trigger_px:
                return ((entry-trigger_px)/entry)-fee, "tp_giveback", i-si+1, best
        low_roi = ((entry-lo)/entry) * lev
        if low_roi > best: best = low_roi
        if best >= tp: armed = True
    if si >= len(d): return -fee, "missing_future", 0, best
    j = min(len(d)-1, max(si, end-1)); ex = float(d.iloc[j].close)
    return ((entry-ex)/entry)-fee, "timeout", j-si+1, best

def _entry_quality(d: pd.DataFrame, si: int, entry: float, obs_min: int = 60, fav: float = 0.006, adv: float = 0.0075) -> Dict[str, Any]:
    end = min(len(d), si + max(1, int(obs_min)))
    if si >= len(d) or end <= si:
        return {"entry_quality_hit": None, "entry_quality_mfe_pct": None, "entry_quality_mae_pct": None}
    w = d.iloc[si:end]
    lows = pd.to_numeric(w["low"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(w["high"], errors="coerce").to_numpy(float)
    mfe = float(np.nanmax(1.0 - lows / float(entry)))
    mae = float(np.nanmax(highs / float(entry) - 1.0))
    hit = bool(mfe >= float(fav) and mae <= float(adv))
    return {
        "entry_quality_hit": hit,
        "entry_quality_mfe_pct": mfe,
        "entry_quality_mae_pct": mae,
        "entry_quality_observation_min": int(obs_min),
        "entry_quality_favorable_pct": float(fav),
        "entry_quality_max_adverse_pct": float(adv),
    }

def _entry_hazard_labels(d: pd.DataFrame, si: int, entry: float) -> Dict[str, Any]:
    """Low-cost labels using already-fetched future 1m bars; starts next full minute."""
    out: Dict[str, Any] = {"schema_version": "entry_hazard_labels.v1"}
    if si >= len(d) or entry <= 0:
        out["available"] = False; return out
    out["available"] = True
    for w in (1,3,5):
        end=min(len(d),si+w); z=d.iloc[si:end]
        if z.empty: continue
        hi=float(pd.to_numeric(z.high,errors="coerce").max())
        lo=float(pd.to_numeric(z.low,errors="coerce").min())
        out[f"mae_{w}m_pct"] = float(hi/entry-1.0)
        out[f"mfe_{w}m_pct"] = float(1.0-lo/entry)
    fav_px=entry*(1.0-0.004); adv_px=entry*(1.0+0.004)
    state="NEITHER"
    for j in range(si,min(len(d),si+5)):
        hi=float(d.iloc[j].high); lo=float(d.iloc[j].low)
        fav=lo<=fav_px; adv=hi>=adv_px
        if fav and adv: state="BOTH_SAME_BAR"; break
        if fav: state="FAVORABLE_FIRST"; break
        if adv: state="ADVERSE_FIRST"; break
    out["touch_40bp_5m"] = state
    return out


def main() -> None:
    src = os.getenv("SHORT_SHADOW_STREAM", "telemetry:short_model_shadow")
    out = os.getenv("SHORT_SHADOW_OUTCOME_STREAM", "telemetry:short_model_shadow_outcomes")
    done_key = os.getenv("SHORT_SHADOW_FINALIZED_SET", "telemetry:short_model_shadow:finalized")
    horizon = max(30, _ei("SHORT_SHADOW_OUTCOME_HORIZON_MIN", 180))
    lev = max(1.0, _ef("SHORT_SHADOW_LEVERAGE", 2.0))
    tp = max(0.0, _ef("FIXED_ROI_TP_PCT", 0.0055)); gb = max(0.0, _ef("FIXED_ROI_TP_GIVEBACK", 0.0008))
    hard = min(0.0, _ef("ROI_HARD_STOP_PCT", -0.0060)); fee = max(0.0, _ef("SHORT_ESTIMATED_ROUND_TRIP_FEE_PCT", 0.0008))
    maxlen = max(1000, _ei("SHORT_SHADOW_OUTCOME_MAXLEN", 20000)); scan_count = max(100, _ei("SHORT_SHADOW_FINALIZER_SCAN_COUNT", 5000))

    bus = RedisBus()
    if not bus.ping(): raise SystemExit("SHORT_SHADOW_FINALIZER: Redis unavailable")
    cutoff = time.time() - (horizon + 2) * 60.0
    pending: List[Tuple[str,Dict[str,Any]]] = []
    for sid, fields in reversed(bus.r.xrevrange(src, max="+", min="-", count=scan_count)):
        try:
            obj = json.loads((fields or {}).get("json") or "{}")
            eid = str(obj.get("event_id") or "")
            if not eid or bus.r.sismember(done_key, eid): continue
            if float(obj.get("ts") or 0.0) > cutoff: continue
            entry = float((obj.get("confirmation") or {}).get("curr_close") or 0.0)
            if entry <= 0.0: continue
            pending.append((sid, obj))
        except Exception: continue
    if not pending:
        print("[SHORT-SHADOW-FINALIZER] pending=0", flush=True); return

    by_symbol: Dict[str,List[Tuple[str,Dict[str,Any]]]] = {}
    for item in pending: by_symbol.setdefault(str(item[1].get("symbol") or "").upper(), []).append(item)
    session = requests.Session(); wrote = 0
    for symbol, items in by_symbol.items():
        ts = [float(x[1].get("ts") or 0.0) for x in items]
        start_ms = (int(min(ts)*1000)//60_000)*60_000 + 60_000
        end_ms = (int(max(ts)*1000)//60_000)*60_000 + (horizon+1)*60_000
        d = _fetch(session, symbol, start_ms, end_ms)
        if d.empty: continue
        opens = pd.to_numeric(d.open_time).astype("int64").to_numpy()
        for source_id, obj in items:
            entry = float((obj.get("confirmation") or {}).get("curr_close") or 0.0)
            minute_ms = (int(float(obj.get("ts") or 0.0)*1000)//60_000)*60_000
            si = int(np.searchsorted(opens, minute_ms+60_000, side="left"))
            if si >= len(d): continue
            net, outcome, mins, best = _sim(d, si, entry, lev, tp, gb, hard, fee, horizon)
            quality = _entry_quality(d, si, entry, obs_min=60, fav=0.006, adv=0.0075)
            hazard_labels = _entry_hazard_labels(d, si, entry)
            result = {"event_id":str(obj.get("event_id") or ""),"source_stream_id":source_id,"source_ts":float(obj.get("ts") or 0.0),"finalized_ts":float(time.time()),"symbol":symbol,"model_provider":str(obj.get("model_provider") or ""),"model_target":str(obj.get("model_target") or ""),"model_feature_hash":str(obj.get("model_feature_hash") or ""),"economic_probability":float(obj.get("economic_probability") or 0.0),"reference_threshold":float(obj.get("reference_threshold") or 0.0),"confirmation_state":str(obj.get("confirmation_state") or ""),"entry_price":entry,"net_return_pct":float(net),"net_return_bps":float(net*10000.0),"outcome":outcome,"exit_minutes":int(mins),"best_roi":float(best),"fee_pct":float(fee),"lifecycle_proxy":{"leverage":lev,"tp_roi":tp,"giveback_roi":gb,"hard_stop_roi":hard,"horizon_min":horizon},"entry_hazard_snapshot":obj.get("entry_hazard_snapshot") or {},"entry_hazard_labels":hazard_labels,**quality}
            out_id = bus.xadd_json(out, result, maxlen=maxlen)
            if out_id:
                bus.r.sadd(done_key, result["event_id"]); wrote += 1
    print(f"[SHORT-SHADOW-FINALIZER] pending={len(pending)} wrote={wrote} symbols={len(by_symbol)}", flush=True)

if __name__ == "__main__":
    main()
