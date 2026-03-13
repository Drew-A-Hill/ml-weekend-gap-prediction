"""
File: price_data_retrieval.py
Author: Drew Hill
This file is used for assembling all the needed price data for feature selection.
"""
from typing import Any, Generator

import pandas as pd
import yfinance as yf

import config

def get_price_history_by_ticker(ticker: yf.Ticker, period: str, interval: str) -> pd.DataFrame:
    """
    Retrieves price details for a given ticker. This data includes Open, High, Low, Close and Volume.
    :param ticker:
    :return:
    """
    data: pd.DataFrame = ticker.history(period=period, interval=interval)

    return data

def _get_list_of_req_metrics(sig: dict[str, Any]) -> list[str]:
    """
    Helper method for that retrieves the list of required metrics for a ticker as marked in the parameters of the
    method signature.
    :param sig: Parameters and input as a dictionary.
    :return: list that contains the required metrics for a ticker.
    """
    include_list: list[str] = []
    count: int = 0

    for param, input_bool in sig.items():
        if isinstance(input_bool, bool):
            if input_bool:
                include_list.append(param)

            else:
                count += 1

    if count == 7:
        return list(sig.keys())

    else:
        return include_list

def build_custom_single_price_df(
        ticker: yf.Ticker,
        period: str,
        interval: str,
        open_p: bool | None = None,
        close_p: bool | None = None,
        high_p: bool | None = None,
        low_p: bool | None = None,
        volume: bool | None = None,
        dividends: bool | None = None,
        stock_splits: bool | None = None
) -> pd.DataFrame:
    """
    Builds a dataframe containing only the required metrics for a ticker. IF NO METRICS ARE MARKED TRUE THEN RETRIEVES
    DATA FOR ALL METRICS.
    :param ticker: ticker object for data to be retrieved and included.
    :param period: period of data to be retrieved and included.
    :param interval: interval of data to be retrieved and included.
    :param open_p: Opening price metric.
    :param close_p: Closing price metric.
    :param high_p: High price metric.
    :param low_p: Low price metric.
    :param volume: Volume metric.
    :param dividends: Dividends metric.
    :param stock_splits: Stock splits metric.
    :return: pd.DataFrame
    """
    price_data: pd.DataFrame = get_price_history_by_ticker(ticker, period, interval)
    include_list: list[str] = _get_list_of_req_metrics(locals())

    df: pd.DataFrame = pd.DataFrame()
    info: dict[str, Any] = ticker.info

    df["Date"] = price_data.index
    df["Year"] = price_data.index.year
    df["Quarter"] = price_data.index.quarter
    df["Ticker"] = ticker.ticker

    for metric in include_list:
        df.set_index("Date", inplace=True)
        df[config.FINANCIAL_METRICS[metric]] = price_data[config.FINANCIAL_METRICS[metric]]

    return df