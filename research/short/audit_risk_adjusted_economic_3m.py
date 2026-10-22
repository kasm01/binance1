#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features.pipeline import FEATURE_SCHEMA_20, make_matrix
from research.short.economic_labels import (
    ShortLifecycleSpec,
    build_short_economic_labels,
)

SYMS = [x.strip().upper() for x in os.getenv("SHORT_ECON_SYMBOLS", "LINKUSDT,AVAXUSDT").split(",") if x.strip()]
SRC = Path(os.getenv("SHORT_ECON_DATA_DIR", "/home/kasm920/binance1-pro/data/offline_cache"))
OUT = Path("research/short/results")
OUT.mkdir(parents=True, exist_ok=True)

SPEC = ShortLifecycleSpec(
    tp_price_pct=0.0055 / 2.0,
    sl_price_pct=0.0060 / 2.0,
    horizon_bars=6,
    round_trip_fee_pct=0.0008,
    conservative_tie_stop=True,
)
FRACS = [0.0025, 0.005, 0.01, 0.02, 0.05]
RISK_LAMBDAS = [0.0, 0.25, 0.50, 1.0, 2.0]


def fit_reg(X, y, seed):
    return HistGradientBoostingRegressor(
        max_iter=140,
        max_leaf_nodes=15,
        l2_regularization=4.0,
        learning_rate=0.05,
        random_state=seed,
    ).fit(X, y)


def metrics(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    if not n:
        return dict(n=0, net_bps=np.nan, win=np.nan, pf=np.nan, sum_ret=0.0)
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    return dict(
        n=n,
        net_bps=float(r.mean() * 1e4),
        win=float((r > 0).mean()),
        pf=float(pos / max(neg, 1e-12)),
        sum_ret=float(r.sum()),
    )


all_folds = []
finals = []

for sym in SYMS:
    path = SRC / f"{sym}_3m_360d.csv"
    if not path.exists():
        print("SKIP_NO_DATA", sym, path, flush=True)
        continue

    d = pd.read_csv(path)
    Xall = make_matrix(d, FEATURE_SCHEMA_20)
    valid = np.isfinite(Xall).all(axis=1)
    valid[-(SPEC.horizon_bars + 2):] = False
    candidate_rows = np.flatnonzero(valid)

    labels = build_short_economic_labels(d, rows=candidate_rows, spec=SPEC)
    if labels.empty:
        print("SKIP_NO_LABELS", sym, flush=True)
        continue

    rows = labels["row"].to_numpy(dtype=int)
    X = Xall[rows]
    net = labels["net_return_pct"].to_numpy(dtype=float)
    mae = labels["mae_pct"].to_numpy(dtype=float)
    mfe = labels["mfe_pct"].to_numpy(dtype=float)

    ok = np.isfinite(X).all(axis=1) & np.isfinite(net) & np.isfinite(mae) & np.isfinite(mfe)
    X, net, mae, mfe = X[ok], net[ok], mae[ok], mfe[ok]
    labels = labels.iloc[np.flatnonzero(ok)].reset_index(drop=True)

    n = len(net)
    hold_start = int(n * 5 / 6)
    dev = np.arange(hold_start)
    hold = np.arange(hold_start, n)
    edges = np.linspace(0, hold_start, 6, dtype=int)

    print("DATA", sym, "n", n, "features", X.shape[1], flush=True)

    fold_rows = []
    for seg in [2, 3, 4]:
        tr = np.arange(0, edges[seg - 1])
        va = np.arange(edges[seg - 1], edges[seg])
        te = np.arange(edges[seg], edges[seg + 1])

        m_ret = fit_reg(X[tr], net[tr], 11000 + seg)
        m_mae = fit_reg(X[tr], mae[tr], 12000 + seg)
        p_ret_v = m_ret.predict(X[va])
        p_mae_v = np.maximum(0.0, m_mae.predict(X[va]))
        p_ret_t = m_ret.predict(X[te])
        p_mae_t = np.maximum(0.0, m_mae.predict(X[te]))

        for lam in RISK_LAMBDAS:
            sv = p_ret_v - lam * p_mae_v
            st = p_ret_t - lam * p_mae_t
            for frac in FRACS:
                th = float(np.quantile(sv, 1.0 - frac))
                z = st >= th
                mm = metrics(net[te][z])
                rec = dict(symbol=sym, fold=seg, risk_lambda=lam, frac=frac, threshold=th, **mm)
                fold_rows.append(rec)
                all_folds.append(rec)

    F = pd.DataFrame(fold_rows)
    choices = []
    for (lam, frac), g in F.groupby(["risk_lambda", "frac"]):
        gg = g[g.n > 0]
        N = int(gg.n.sum())
        posfold = int((gg.net_bps > 0).sum())
        weighted = float(np.average(gg.net_bps, weights=gg.n)) if N else np.nan
        worst = float(gg.net_bps.min()) if len(gg) else np.nan
        choices.append(dict(risk_lambda=lam, frac=frac, wf_n=N, wf_posfold=posfold, wf_net_bps=weighted, wf_worst_bps=worst))

    C = pd.DataFrame(choices).sort_values(
        ["wf_posfold", "wf_net_bps", "wf_worst_bps", "wf_n"],
        ascending=[False, False, False, False],
    )
    eligible = C[(C.wf_posfold >= 2) & (C.wf_n >= 30)]
    chosen = eligible.iloc[0] if len(eligible) else C.iloc[0]
    lam = float(chosen.risk_lambda)
    frac = float(chosen.frac)

    m_ret = fit_reg(X[dev], net[dev], 19001)
    m_mae = fit_reg(X[dev], mae[dev], 19002)
    sv = m_ret.predict(X[dev]) - lam * np.maximum(0.0, m_mae.predict(X[dev]))
    sh = m_ret.predict(X[hold]) - lam * np.maximum(0.0, m_mae.predict(X[hold]))
    th = float(np.quantile(sv, 1.0 - frac))
    z = sh >= th
    mm = metrics(net[hold][z])
    selected_labels = labels.iloc[hold[z]]

    res = dict(
        symbol=sym,
        risk_lambda=lam,
        frac=frac,
        wf_posfold=int(chosen.wf_posfold),
        wf_n=int(chosen.wf_n),
        wf_net_bps=float(chosen.wf_net_bps),
        wf_worst_bps=float(chosen.wf_worst_bps),
        threshold=th,
        hold_tp=int((selected_labels.outcome == "tp").sum()),
        hold_sl=int((selected_labels.outcome == "sl").sum()),
        hold_timeout=int((selected_labels.outcome == "timeout").sum()),
        hold_mean_mae_bps=float(selected_labels.mae_pct.mean() * 1e4) if len(selected_labels) else np.nan,
        hold_mean_mfe_bps=float(selected_labels.mfe_pct.mean() * 1e4) if len(selected_labels) else np.nan,
        hold_mean_hit_bar=float(selected_labels.hit_bar.mean()) if len(selected_labels) else np.nan,
        **{f"hold_{k}": v for k, v in mm.items()},
    )
    finals.append(res)
    print("RESULT", json.dumps(res, sort_keys=True), flush=True)

pd.DataFrame(all_folds).to_csv(OUT / "SHORT_RISK_ADJ_ECON_3M_WF.csv", index=False)
pd.DataFrame(finals).to_csv(OUT / "SHORT_RISK_ADJ_ECON_3M_HOLDOUT.csv", index=False)
(OUT / "SHORT_RISK_ADJ_ECON_3M_HOLDOUT.json").write_text(json.dumps(finals, indent=2))
print("FINAL")
print(pd.DataFrame(finals).to_string(index=False))
