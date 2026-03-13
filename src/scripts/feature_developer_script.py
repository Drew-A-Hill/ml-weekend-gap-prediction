"""

"""
import pandas as pd
import yfinance as yf

import src.data_pipelines.featured_companies.featured_company_retrieval as featured
from src import config

all_tickers: pd.Series = featured.full_list_of_tickers()
m_cap_min: int = 10_000_000_000
m_cap_max: int = 200_000_000_000

def dev_featured_companies() -> :
    """

    :return:
    """
    featured_companies_list: list[yf.Ticker] = []

    for ticker_str in all_tickers:
        ticker: yf.Ticker = yf.Ticker(ticker_str)

        if (featured.check_companies_by_sector(ticker, config.INDUSTRIES)
            and featured.check_companies_by_market_cap(ticker, m_cap_min, m_cap_max)
            and featured.check_companies_by_profitability(ticker, .05)
        ):
            yield ticker

        else:
            continue
