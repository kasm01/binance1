import numpy as np

import logging


logger = logging.getLogger(__name__)

def safe_p_buy(model, X) -> float:
    """
    Online modelden p_buy üret:
      1) predict_proba (tercih)
      2) decision_function -> sigmoid
      3) fallback predict (0/1)

    Notlar:
      - robust: exception olursa fallback’a iner
      - clamp: NaN/inf/taşma durumlarını güvenli aralıkta tutar
      - debug: 60 sn’de bir gerçek p_buy’ı ve X stats basar
    """
    # --- DISABLE_SGD OnlineLearner guard ---
    try:
        import os
        _disable_sgd = str(os.getenv("DISABLE_SGD", "0")).lower() in ("1", "true", "yes", "on")
        if _disable_sgd and model.__class__.__name__ == "OnlineLearner":
            return 0.5
    except Exception:
        pass

def _clamp01(v: float) -> float:
    if not np.isfinite(v):
        return 0.5
    # saturation guard
    if v < 0.05:
        return 0.05
    if v > 0.95:
        return 0.95
    return float(v)

    # --- compute p safely ---
    p = None
    src = "UNKNOWN"
    err = None

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # proba shape: (n, 2) bekliyoruz
            p = float(proba[0, 1])
            src = "predict_proba"
        except Exception as e:
            err = f"predict_proba:{e}"

    if p is None and hasattr(model, "decision_function"):
        try:
            z = float(model.decision_function(X)[0])
            # sigmoid
            p = float(1.0 / (1.0 + np.exp(-z)))
            src = "decision_function"
        except Exception as e:
            err = (err + " | " if err else "") + f"decision_function:{e}"

    if p is None:
        try:
            pred = model.predict(X)[0]
            p = 1.0 if int(pred) == 1 else 0.0
            src = "predict"
        except Exception as e:
            # en kötü durumda bile botu düşürmeyelim
            logger.exception("[safe_p_buy] predict fallback failed: %s", e)
            return 0.5

    p = _clamp01(p)

    # --- debug throttle 60s ---
    try:
        import time
        now = time.time()
        last = globals().get("_SAFE_P_BUY_DBG_LAST_TS", 0) or 0
        if (now - float(last)) > 60:
            globals()["_SAFE_P_BUY_DBG_LAST_TS"] = now
            Xa = np.asarray(X) if X is not None else None
            x_min = float(np.nanmin(Xa)) if Xa is not None else None
            x_max = float(np.nanmax(Xa)) if Xa is not None else None
            x_nan = int(np.isnan(Xa).sum()) if Xa is not None else None
            cls = type(model).__name__ if model is not None else None
            loss = getattr(model, "loss", None)
            logger.info(
                "[SAFE_P_BUY_DBG] model=%s loss=%s src=%s p_buy=%.10f X[min,max,nan]=%r %r %r err=%r",
                cls, loss, src, p, x_min, x_max, x_nan, err
            )
    except Exception:
        pass

    return p

