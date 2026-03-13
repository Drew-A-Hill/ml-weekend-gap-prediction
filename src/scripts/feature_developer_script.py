"""
File: feature_developer_script.py
Author: Drew Hill
This script is used to assemble the data to be used in feature selection process.
"""
from typing import Generator

import pandas as pd
import yfinance as yf

import data_pipelines.featured_companies.featured_company_retrieval as featured
from src import config

def dev_featured_companies() -> Generator[yf.Ticker, None]:
    """
    Develops the companies to be used in the model based on the criteria and yields each yf.Ticker object.
    :Yields: yf.Ticker objects
    """
    all_tickers: pd.Series = featured.full_list_of_tickers()
    m_cap_min: int = 10_000_000_000
    m_cap_max: int = 200_000_000_000

    for ticker_str in all_tickers:
        ticker: yf.Ticker = yf.Ticker(ticker_str)

        if (featured.check_companies_by_sector(ticker, config.INDUSTRIES)
            and featured.check_companies_by_market_cap(ticker, m_cap_min, m_cap_max)
            and featured.check_companies_by_profitability(ticker, .05)
        ):
            yield ticker

        else:
            continue

def dev_data(ticker: yf.Ticker) -> pd.DataFrame:
    """

    :param ticker:
    :return:
    """
    pass

