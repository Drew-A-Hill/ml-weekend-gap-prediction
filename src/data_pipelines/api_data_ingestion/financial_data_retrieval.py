"""

"""
import pandas as pd
import yfinance as yf
from yfinance import Ticker


def add_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """

    :param data:
    :return:
    """
    tp: int = (data["High"] + data["Low"] + data["Close"]) / 3
    data["vwap"] = (tp * data["Volume"]).cumsum() / data["Volume"].cumsum()

    return data

def add_weekly_avg(data: pd.DataFrame) -> pd.DataFrame:
    """

    :param data:
    :return:
    """
    data["weekly_avg_vol"] = data["Volume"].rolling(5).mean()

    return data

def get_price_info(ticker: Ticker) -> pd.DataFrame:
    """

    :param ticker:
    :return:
    """
    data: pd.DataFrame = ticker.history(period="10y", interval="1d")
    data["ticker"] = ticker.ticker

    data = add_vwap(data)


    return data