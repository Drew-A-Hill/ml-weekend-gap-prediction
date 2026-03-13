"""
File: feature_developer_script.py
Author: Drew Hill
This script is used to assemble the data to be used in feature selection process.
"""
from typing import Generator, Any

import pandas as pd
import yfinance as yf

import data_pipelines.featured_companies.featured_company_retrieval as featured
import data_pipelines.api_data_ingestion.price_data_retrieval as price_data
from src import config

def dev_featured_companies() -> Generator[yf.Ticker, None] | None:
    """
    Develops the companies to be used in the model based on the criteria and yields each yf.Ticker object.
    :Yields: yf.Ticker objects
    """
    all_tickers: pd.Series = featured.full_list_of_tickers()
    m_cap_min: int = 10_000_000_000
    m_cap_max: int = 200_000_000_000

    count = 0
    list_t : list[str] = []
    for ticker_str in all_tickers:
        ticker: yf.Ticker = yf.Ticker(ticker_str)
        company_info: dict[str, Any] = ticker.info
        if count < 80:
            if (featured.check_companies_by_industry(company_info, config.INDUSTRIES)
                and featured.check_companies_by_market_cap(company_info, m_cap_min, m_cap_max)
                and featured.check_companies_by_profitability(company_info, .05)
            ):
                list_t.append(ticker_str)
                print(f"Count: {count} Ticker: {ticker_str}")
                count += 1
                yield ticker

            else:
                print(f"Count: {count}")
                print(False)
                count += 1
                continue

        else:

            return


def dev_data() -> pd.DataFrame:
    """
    Develops dataframe with data for feature selection.
    :return: pd.DataFrame
    """
    df: pd.DataFrame = pd.DataFrame()

    for company in dev_featured_companies():
        new_df: pd.DataFrame = price_data.build_custom_single_price_df(company, "10y", "1d", open_p=True)
        df = pd.concat([df, new_df])

    return df

def main():
    # print(featured.full_list_of_tickers())
    print(dev_data())

if __name__ == "__main__":
    main()

