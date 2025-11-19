import time
import functools
from typing import Callable, Tuple, Type

from core.exceptions import RetryLimitExceeded


def retry(
    _func: Callable | None = None,
    *,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    tries: int | None = None,
    max_attempts: int | None = None,
    delay: float = 1.0,
    backoff: float = 1.0,
) -> Callable:
    """
    Esnek retry decorator'u.

    Destekler:
        @retry(max_attempts=5, delay=2, exceptions=(...))
        @retry(exceptions=(...), tries=3, delay=2, backoff=2)
        @retry  (parametresiz kullanım)
    """

    if tries is None and max_attempts is None:
        total_tries = 3
    else:
        total_tries = tries if tries is not None else max_attempts

    def decorator_retry(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries = total_tries
            _delay = delay
            attempt = 0

            while _tries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    _tries -= 1

                    if _tries <= 0:
                        raise RetryLimitExceeded(
                            f"{func.__name__} retry limit exceeded after {attempt} attempts"
                        ) from e

                    print(
                        f"[retry] {func.__name__} exception: {e}. "
                        f"Retrying {attempt}/{total_tries} after {_delay} seconds..."
                    )
                    time.sleep(_delay)
                    _delay *= backoff

        return wrapper

    if _func is not None:
        return decorator_retry(_func)

    return decorator_retry
