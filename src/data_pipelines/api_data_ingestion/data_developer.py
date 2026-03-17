"""
File: data_developer.py
Author: Drew Hill
This file is used to develop the data set to be used for the model.
"""
import time
from urllib.error import HTTPError
import pandas as pd
import yfinance as yf
import utils.df_expansion as merge

import data_pipelines.api_data_ingestion.fundamentals_data_retrieval as fd
from yfinance.exceptions import YFRateLimitError
import data_pipelines.api_data_ingestion.price_data_retrieval as price_data
import data_pipelines.featured_companies.company_retrieval as featured
import data_pipelines.featured_companies.company_filters_on_call as filters
from data_io.read_write_data import read_from_csv
from data_pipelines.featured_companies.company_retrieval import get_cik
from utils.general import ticker_iter_w_progress

def dev_featured_companies(
        by_industry: bool = False,
        by_market_cap: bool = False,
        by_profitability: bool = False,
        by_public_age: bool = False
) -> pd.Series:
    """
    Develops the companies to be used in the model based on the criteria.
    return: panda series that contains a tuple for each company with ticker object, cik, and ticker_str.
    """
    featured_companies: list[str] = []

    for ticker in ticker_iter_w_progress("Filtering Companies", featured.get_all_tickers().index):
        try:
            if filters.filter_on_call(
                    ticker,
                    by_industry=by_industry,
                    by_market_cap=by_market_cap,
                    by_profitability=by_profitability,
                    by_public_age=by_public_age,
            ):
                featured_companies.append(ticker)

        except YFRateLimitError:
            time.sleep(300)

        except HTTPError:
            continue

        except KeyboardInterrupt:
            raise

    return pd.Series(featured_companies, name="featured_tickers")

def dev_price_data_by_ticker(df: pd.DataFrame, ticker: yf.Ticker) -> pd.DataFrame:
    """
    Develops dataframe with price dataset to be used for feature selection.
    :returns: dataframe with price data
    """
    df: pd.DataFrame = df

    new_df: pd.DataFrame = price_data.build_single_ticker_price_df(
        ticker, open_p=True, close_p=True, high_p=True, low_p=True, volume=True)

    df = pd.concat([df, new_df])

    return df

def dev_fundamental_data_by_ticker(df: pd.DataFrame, cik: int, ticker_str) -> pd.DataFrame:
    """
    Develops dataframe with fundamental dataset to be used for feature selection.
    :returns: dataframe with fundamental data
    """
    df: pd.DataFrame = df
    try:
        new_df: pd.DataFrame = fd.build_single_ticker_fundamentals_df(cik, ticker_str, 10)
        df = pd.concat([df, new_df])

    except ValueError:
        pass

    return df

def dev_price_and_fundamental_data_by_ticker(companies: pd.DataFrame) -> pd.DataFrame:
    """

    """
    p_df: pd.DataFrame = pd.DataFrame()
    f_df: pd.DataFrame = pd.DataFrame()

    for ticker_str in ticker_iter_w_progress("Collect Data", companies["featured_tickers"]):

        p_df = dev_price_data_by_ticker(p_df, yf.Ticker(ticker_str))
        f_df = dev_fundamental_data_by_ticker(f_df, get_cik(ticker_str), ticker_str)

    df = merge.merge_df_columns([p_df, f_df])

    return df
