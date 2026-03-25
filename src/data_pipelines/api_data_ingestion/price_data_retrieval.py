"""
File: price_data_retrieval.py
Author: Drew Hill

This file builds a dataframe containing the indicated price data for a given ticker.

Usage:
    get_price_data(open_p=True, close_p=True, high_p=True, low_p=True, volume=True)
    get_price_data(open_p=False, close_p=False, high_p=False)
"""
import pandas as pd
import yfinance as yf
import config
import utils.pipline_helpers as helpers
from data_pipelines.api_clients.yahoo_client import get_price_history

def get_price_data(
        ticker_str: str,
        open_p: bool = False,
        close_p: bool = False,
        high_p: bool = False,
        low_p: bool = False,
        volume: bool = False,
        dividends: bool = False,
        stock_splits: bool = False
) -> pd.DataFrame:
    """
    Builds a dataframe containing only the required metrics for a ticker. IF NO METRICS ARE MARKED TRUE THEN RETRIEVES
    DATA FOR ALL METRICS.
    :param ticker_str: ticker symbol for data to be retrieved and included.
    :param open_p: Opening price metric.
    :param close_p: Closing price metric.
    :param high_p: High price metric.
    :param low_p: Low price metric.
    :param volume: Volume metric.
    :param dividends: Dividends metric.
    :param stock_splits: Stock splits metric.
    :return: pd.DataFrame
    """
    price_data: pd.DataFrame = get_price_history(
        ticker_str,
        period=config.PERIOD,
        interval=config.INTERVAL
    ).copy()

    price_data.index = pd.to_datetime(price_data.index)

    include_list: list[str] = helpers.get_list_of_req_metrics(locals())
    ticker: yf.Ticker = yf.Ticker(ticker_str)

    df: pd.DataFrame = pd.DataFrame()

    df["Year"] = price_data.index.year
    df["Quarter"] = price_data.index.quarter.map(lambda q: f"Q{q}")
    df["Date"] = price_data.index

    for metric in include_list:
        df["Ticker"] = ticker.ticker
        df[config.FINANCIAL_METRICS[metric]] = price_data[
            config.FINANCIAL_METRICS[metric]
        ].values

    return df