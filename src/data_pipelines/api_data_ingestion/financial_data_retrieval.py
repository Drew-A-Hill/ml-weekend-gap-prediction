"""
File: financial_data_retrieval.py
Author: Drew Hill
This file is used for assembling all the needed price data for feature selection.
"""
import pandas as pd
from yfinance import Ticker

def get_price_history_by_ticker(ticker: Ticker, period: str, interval: str) -> pd.DataFrame:
    """
    Retrieves price details for a given ticker. This data includes Open, High, Low, Close and Volume.
    :param ticker:
    :return:
    """
    data: pd.DataFrame = ticker.history(period=period, interval=interval)

    # Adds ticker as first col in dataframe
    data["ticker"] = ticker.ticker

    columns = data["ticker"] + [c for c in data.columns if c != "ticker"]

    return data[columns]