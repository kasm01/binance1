"""
data/feature_engineer.py

Eski import yolunu (data.feature_engineer) yeni modul (data.feature_engineering)
üzerinden çalıştırmak için küçük bir uyumluluk katmanı (shim).

Örnek eski kullanım:
    from data.feature_engineer import FeatureEngineer

Bu dosya, aslında data.feature_engineering içindeki FeatureEngineer sınıfını export eder.
"""

from .feature_engineering import FeatureEngineer  # type: ignore

__all__ = ["FeatureEngineer"]

