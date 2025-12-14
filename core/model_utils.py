import numpy as np

def safe_p_buy(model, X) -> float:
    """
    Online modelden p_buy üret:
      1) predict_proba varsa direkt
      2) decision_function varsa sigmoid
      3) fallback predict (0/1)
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0, 1]
        return float(p)

    if hasattr(model, "decision_function"):
        z = float(model.decision_function(X)[0])
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p)

    pred = model.predict(X)[0]
    return 1.0 if int(pred) == 1 else 0.0
