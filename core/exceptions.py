"""
Custom exception classes for the Binance1-Pro bot.

Bu dosyadaki sınıflar, projenin farklı katmanlarındaki
hataları daha anlamlı bir şekilde yakalamak ve loglamak için kullanılır.
"""


class ConfigException(Exception):
    """Environment / config yükleme ile ilgili hatalar."""
    pass


class DataLoadingException(Exception):
    """Veri indirme / yükleme sırasında oluşan hatalar."""
    pass


class FeatureEngineeringException(Exception):
    """Feature engineering aşamasında oluşan hatalar."""
    pass


class LabelGenerationException(Exception):
    """Label (hedef değişken) oluştururken oluşan hatalar."""
    pass


class DataProcessingException(Exception):
    """
    Genel veri işleme hataları.
    Örneğin, pipeline içindeki ara adımların beklenmedik çıktıları vb.
    """
    pass


class ModelTrainingException(Exception):
    """Batch veya diğer modelleri eğitirken oluşan hatalar."""
    pass


class OnlineLearningException(Exception):
    """OnlineLearner ile ilgili hatalar (initial_fit, partial_update vb.)."""
    pass


class SignalGenerationException(Exception):
    """Sinyal üretimi sırasında oluşan hatalar."""
    pass


class CacheException(Exception):
    """Redis / cache katmanındaki hatalar."""
    pass
