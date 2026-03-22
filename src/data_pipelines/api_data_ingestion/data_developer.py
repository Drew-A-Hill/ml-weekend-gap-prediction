"""
File: data_developer.py
Author: Drew Hill
This file is used to develop the data set to be used for the model.
"""
import pandas as pd
import yfinance as yf
import utils.df_expansion as merge
import data_pipelines.api_data_ingestion.fundamentals_data_retrieval as fd
import data_pipelines.api_data_ingestion.price_data_retrieval as price_data
from data_pipelines.company_selection.registered_companies import get_cik
from utils.terminal_run_status import ticker_iter_w_progress

# def dev_price_data_by_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
#     """
#     Develops dataframe with price dataset to be used for feature selection.
#     :returns: dataframe with price data
#     """
#     df: pd.DataFrame = df
#
#     new_df: pd.DataFrame = price_data.build_single_ticker_price_df(
#         ticker, open_p=True, close_p=True, high_p=True, low_p=True, volume=True)
#
#     df = pd.concat([df, new_df])
#
#     return df
#
# def dev_fundamental_data_by_ticker(df: pd.DataFrame, cik: int, ticker_str) -> pd.DataFrame:
#     """
#     Develops dataframe with fundamental dataset to be used for feature selection.
#     :returns: dataframe with fundamental data
#     """
#     df: pd.DataFrame = df
#     try:
#         new_df: pd.DataFrame = fd.build_single_ticker_fundamentals_df(cik, ticker_str, 10)
#         df = pd.concat([df, new_df])
#
#     except ValueError:
#         pass
#
#     return df
#
# def dev_price_and_fundamental_data_by_ticker(companies: pd.DataFrame) -> pd.DataFrame:
#     """
#
#     """
#     p_df: pd.DataFrame = pd.DataFrame()
#     f_df: pd.DataFrame = pd.DataFrame()
#
#     for ticker_str in ticker_iter_w_progress("Collecting Data", companies["featured_tickers"]):
#
#         p_df = dev_price_data_by_ticker(p_df, yf.Ticker(ticker_str))
#         f_df = dev_fundamental_data_by_ticker(f_df, get_cik(ticker_str), ticker_str)
#
#     df = merge.merge_df_columns([p_df, f_df])
#
#     return df

def dev_dataset_by_ticker(
        price: pd.DataFrame|None = None,
        fundamental: pd.DataFrame|None = None,
        indicators: pd.DataFrame|None = None
) -> pd.DataFrame:
    """

    """
    df: pd.DataFrame = pd.DataFrame()

    if price is not None and not price.empty:
        df = price.copy()

    if fundamental is not None and not fundamental.empty:
        if df.empty:
            df = fundamental.copy()
        else:
            df = pd.merge(df, fundamental, on=["Ticker", "Year", "Quarter"], how="outer")

    if indicators is not None and not indicators.empty:
        if df.empty:
            df = indicators.copy()
        else:
            df = pd.merge(df, indicators, on=["Ticker", "Year", "Quarter"], how="outer")

    return df