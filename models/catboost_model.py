# models/catboost_model.py
import os
from typing import Optional

from core.logger import system_logger

# catboost paketi opsiyonel: lokalde yoksa import patlamasın
try:
    from catboost import CatBoostClassifier  # type: ignore
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None  # type: ignore
    CATBOOST_AVAILABLE = False
    system_logger.warning(
        "[CatBoostModel] 'catboost' paketi yüklü değil. "
        "Lokal ortamda CatBoostModel kullanılamayacak."
    )


class CatBoostModel:
    """
    CatBoost tabanlı model wrapper'ı.

    - Cloud Run / prod ortamında: catboost kurulu, model normal çalışır.
    - Lokal ortamda (Python 3.12, catboost kurulmamış): import başarısız olmaz,
      sadece bu sınıfı kullanmaya çalışırsan RuntimeError fırlatır.
    """

    def __init__(self, model_path: Optional[str] = None):
        if not CATBOOST_AVAILABLE:
            # Import aşamasında patlatmıyoruz; sadece gerçekten kullanmaya kalkınca hata veriyoruz.
            raise RuntimeError(
                "CatBoostModel kullanılamıyor çünkü 'catboost' paketi yüklü değil. "
                "Lütfen sadece LSTM / LightGBM / Fallback modellerini kullanın "
                "veya Python 3.11 ortamında catboost kurulmuş şekilde çalıştırın."
            )

        self.model = CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
        )

        # Eğitimli model dosyası verilmişse yükle
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_model(model_path)
                system_logger.info(f"[CatBoostModel] Model yüklendi: {model_path}")
            except Exception as e:
                system_logger.error(f"[CatBoostModel] Model yüklenemedi: {e}")

    def fit(self, X, y):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoostModel.fit çağrıldı ama 'catboost' paketi yok.")

        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoostModel.predict_proba çağrıldı ama 'catboost' paketi yok.")

        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Binary sınıflandırma için proba[1] > 0.5 ise 1, değilse 0 döner.
        """
        proba = self.predict_proba(X)

        # === SGD_PROBA_SAT_DBG (auto) ===
        try:
            import numpy as _np
            from datetime import datetime as _dt
            _now = _dt.utcnow().timestamp()
            _last = globals().get('_SGD_PROBA_DBG_LAST_TS', 0) or 0
            if (_now - float(_last)) > 60:
                globals()['_SGD_PROBA_DBG_LAST_TS'] = _now
                _mdl = self
                _mname = _mdl.__class__.__name__ if _mdl is not None else None
                if _mname and ('SGD' in _mname or 'SGDClassifier' in _mname):
                    _X = X
                    _Xn = _np.asarray(_X) if _X is not None else None
                    _pn = _np.asarray(proba) if proba is not None else None
                    _nan = int(_np.isnan(_Xn).sum()) if _Xn is not None and _Xn.size else 0
                    _inf = int(_np.isinf(_Xn).sum()) if _Xn is not None and _Xn.size else 0
                    _xmin = float(_np.nanmin(_Xn)) if _Xn is not None and _Xn.size else None
                    _xmax = float(_np.nanmax(_Xn)) if _Xn is not None and _Xn.size else None
                    _pmin = float(_np.nanmin(_pn)) if _pn is not None and _pn.size else None
                    _pmax = float(_np.nanmax(_pn)) if _pn is not None and _pn.size else None
                    _p0 = _pn[0].tolist() if _pn is not None and _pn.ndim==2 and _pn.shape[0]>0 else None
                    _p0max = float(_np.max(_pn[0])) if _pn is not None and _pn.ndim==2 and _pn.shape[0]>0 else None
                    _classes = getattr(_mdl, 'classes_', None)
                    _df_min = _df_max = None
                    try:
                        _df = _mdl.decision_function(_X)
                        _dfn = _np.asarray(_df)
                        _df_min = float(_np.nanmin(_dfn)) if _dfn.size else None
                        _df_max = float(_np.nanmax(_dfn)) if _dfn.size else None
                    except Exception:
                        pass
                    _coef_norm = _inter = None
                    try:
                        _c = getattr(_mdl, 'coef_', None)
                        _b = getattr(_mdl, 'intercept_', None)
                        if _c is not None:
                            _coef_norm = float(_np.linalg.norm(_np.asarray(_c)))
                        if _b is not None:
                            _inter = _np.asarray(_b).tolist()
                    except Exception:
                        pass
                    _log = globals().get('system_logger', None) or globals().get('logger', None)
                    _msg = (
                        f"[SGD_PROBA_DBG] model={_mname} classes={_classes} "
                        f"X_shape={None if _Xn is None else _Xn.shape} nan={_nan} inf={_inf} "
                        f"Xmin={_xmin} Xmax={_xmax} "
                        f"pmin={_pmin} pmax={_pmax} p0={_p0} p0max={_p0max} "
                        f"df_min={_df_min} df_max={_df_max} coef_norm={_coef_norm} intercept={_inter}"
                    )
                    try:
                        (_log.info if _log else print)(_msg)
                    except Exception:
                        print(_msg)
        except Exception:
            pass
        # === /SGD_PROBA_SAT_DBG ===
        return (proba[:, 1] > 0.5).astype(int)

