import time
import functools
from core.exceptions import RetryLimitExceeded

def retry(max_attempts=3, delay=2, exceptions=(Exception,)):
    """
    Fonksiyon çağrısı başarısız olursa retry mekanizması uygular.
    Args:
        max_attempts (int): Maksimum deneme sayısı.
        delay (int | float): Her deneme arasında bekleme süresi (saniye).
        exceptions (tuple): Hangi exception'larda retry yapılacağı.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    print(f"Retry {attempts}/{max_attempts} for {func.__name__} due to {e}")
                    time.sleep(delay)
                    if attempts >= max_attempts:
                        raise RetryLimitExceeded(f"{func.__name__} retry limit exceeded") from e
        return wrapper
    return decorator
