"""

"""
import time

import pandas as pd
import yfinance as yf
from curl_cffi.requests.exceptions import HTTPError

def _rate_limit_exception() -> None:
    """

    :return:
    """
    pass

def get_info(ticker: str) -> dict[str, dict[str, str]] | None:
    """

    :return:
    """
    for attempt in range(5):
        try:
            return yf.Ticker(ticker).info

        except HTTPError:
            return None

        except Exception as e:
            msg = str(e)

            if "Too Many Requests" in msg or "Rate limited" in msg:
                time.sleep(60 * (attempt + 1))  # backoff
                continue

            return None

    return None

def get_fast_info(ticker: str) -> dict[str, dict[str, str]] | None:
    """

    :return:
    """
    count: int = 0

    for attempt in range(5):
        try:
            return yf.Ticker(ticker).fast_info

        except Exception as e:
            msg = str(e)

            if "Too Many Requests" in msg:
                time.sleep(30 * (attempt + 1))
                continue

            return None

    return None

def get_history(ticker: str, period: str, interval: str = None) -> pd.DataFrame | None:
    """

    :return:
    """
    for attempt in range(5):
        try:
            t = yf.Ticker(ticker)

            if interval:
                return t.history(period=period, interval=interval)
            else:
                return t.history(period=period)

        except (HTTPError, ValueError):
            return None

        except Exception as e:
            msg = str(e)

            if "Too Many Requests" in msg:
                time.sleep(60 * (attempt + 1))
                continue

            return None

    return None
