import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from core.logger import system_logger


class BatchLearner:
    """
    Batch eğitim: clean feature DF + target ile LightGBM modeli eğitir.
    Aynı zamanda validation metriklerini log'a yazar.
    """

    def __init__(self, logger=None):
        self.logger = logger or system_logger

    def train(self, X: pd.DataFrame, y: pd.Series) -> LGBMClassifier:
        # Temel istatistikler
        n = len(y)
        n_pos = int(y.sum())
        n_neg = n - n_pos
        pos_ratio = n_pos / n if n > 0 else 0.0

        self.logger.info(
            "[BATCH] Preparing data for training: n=%d, pos=%d, neg=%d, pos_ratio=%.3f",
            n, n_pos, n_neg, pos_ratio
        )

        if n < 200 or n_pos == 0 or n_neg == 0:
            self.logger.warning(
                "[BATCH] Data too small or only one class present. Falling back to simple model."
            )
            # Çok küçük veri ya da tek sınıf varsa bile model eğitiriz ama uyarı veririz
            model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                objective="binary",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X, y)
            self.logger.info("[BATCH] LightGBM training completed (fallback settings).")
            return model

        # Train/Validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # LightGBM modeli (class_weight balanced ile)
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",   # dengesiz veri için önemli
            n_jobs=-1,
            random_state=42,
        )

        self.logger.info("[BATCH] Starting LightGBM training...")
        model.fit(X_train, y_train)

        # Validation tahminleri
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_val, y_proba)
        except Exception:
            auc = float("nan")

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)

        self.logger.info(
            "[BATCH] Validation metrics: AUC=%.4f, ACC=%.4f, PREC=%.4f, REC=%.4f",
            auc, acc, prec, rec
        )

        self.logger.info("[BATCH] LightGBM training completed successfully.")
        return model
