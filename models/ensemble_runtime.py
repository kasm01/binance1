from __future__ import annotations
import os
import numpy as np

from models.lightgbm_runtime import LightGBMRuntime

class EnsembleRuntime:
    def __init__(self, lgb_path: str | None, lstm_predict_fn):
        self.lgb = LightGBMRuntime(lgb_path) if lgb_path else None
        self.lstm_predict_fn = lstm_predict_fn  # fn(X)->p1 array
        self.w_lgb = float(os.getenv("ENS_W_LGB", "0.55"))
        self.w_lstm = float(os.getenv("ENS_W_LSTM", "0.45"))

    def predict_p1(self, X: np.ndarray, debug: dict | None = None) -> np.ndarray:
        debug = debug if isinstance(debug, dict) else {}
        preds = []
        ws = []

        if self.lgb is not None:
            try:
                p_lgb = self.lgb.predict_proba_p1(X)
                preds.append(p_lgb); ws.append(self.w_lgb)
                debug["p_lgb_mean"] = float(np.mean(p_lgb))
            except Exception as e:
                debug["lgb_error"] = str(e)

        try:
            p_lstm = self.lstm_predict_fn(X)
            preds.append(p_lstm); ws.append(self.w_lstm)
            debug["p_lstm_mean"] = float(np.mean(p_lstm))
        except Exception as e:
            debug["lstm_error"] = str(e)

        if not preds:
            p = np.full(X.shape[0], 0.5, dtype=float)
            debug["ens_mode"] = "uniform_fallback"
            return p

        W = np.array(ws, dtype=float)
        W = W / max(W.sum(), 1e-9)
        p = np.zeros(X.shape[0], dtype=float)
        for w, pr in zip(W, preds):
            p += float(w) * np.asarray(pr, dtype=float)

        p = np.clip(p, 0.0, 1.0)
        debug["ens_mode"] = "lgb+lstm" if (self.lgb is not None) else "lstm_only"
        debug["p_ens_mean"] = float(np.mean(p))
        return p
