import numpy as np

def safe_p_buy(model, X) -> float:
    # === SAFE_P_BUY_DBG (auto) ===
    try:
        import numpy as _np
        from datetime import datetime as _dt
        _now = _dt.utcnow().timestamp()
        _last = globals().get('_SAFE_P_BUY_DBG_LAST_TS', 0) or 0
        if (_now - float(_last)) > 60:
            globals()['_SAFE_P_BUY_DBG_LAST_TS'] = _now
            _X = X
            _Xa = _np.asarray(_X) if _X is not None else None
            _x_min = float(_np.nanmin(_Xa)) if _Xa is not None else None
            _x_max = float(_np.nanmax(_Xa)) if _Xa is not None else None
            _x_nan = int(_np.isnan(_Xa).sum()) if _Xa is not None else None
            _cls = type(model).__name__ if model is not None else None
            _has_pp = hasattr(model, 'predict_proba')
            # ham predict_proba snapshot (exception swallow)
            _pp = None
            if _has_pp:
                try:
                    _pp = model.predict_proba(X)
                except Exception as _e:
                    _pp = f'EXC:{_e}'
            print("[SAFE_P_BUY_DBG] model=%s has_predict_proba=%s X[min,max,nan]=%r %r %r proba=%r" % (_cls, _has_pp, _x_min, _x_max, _x_nan, _pp))
    except Exception:
        pass
    # === /SAFE_P_BUY_DBG ===
    """
    Online modelden p_buy üret:
      1) predict_proba varsa direkt
      2) decision_function varsa sigmoid
      3) fallback predict (0/1)
    """
    # --- DISABLE_SGD OnlineLearner guard ---
    try:
        import os
        _disable_sgd = str(os.getenv("DISABLE_SGD","0")).lower() in ("1","true","yes","on")
        if _disable_sgd and model.__class__.__name__ == "OnlineLearner":
            return 0.5
    except Exception:
        pass

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0, 1]
        return float(p)

    if hasattr(model, "decision_function"):
        z = float(model.decision_function(X)[0])
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p)

    pred = model.predict(X)[0]
    return 1.0 if int(pred) == 1 else 0.0
