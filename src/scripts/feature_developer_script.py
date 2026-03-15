"""
File: feature_developer_script.py
Author: Drew Hill
This script is used to assemble the data to be used in feature selection process.
"""
from pathlib import Path
from typing import Generator, Any

import pandas as pd
import yfinance as yf

import data_pipelines.featured_companies.company_retrieval as featured
import data_pipelines.api_data_ingestion.price_data_retrieval as price_data
import data_pipelines.featured_companies.featured_company_filters as filters
from data_pipelines.api_data_ingestion.df_expansion import merge_df_columns
from data_pipelines.api_data_ingestion.fundamentals_data_retrieval import build_single_ticker_fundamentals_df
from data_io import read_write_data as read_write


#TODO MAYBE MOVE THIS TO FILTER AND MAKE THE DATA AN INPUT
def dev_featured_companies() -> Generator[tuple[yf.Ticker, int, str]] | None:
    """
    Develops the companies to be used in the model based on the criteria and yields each yf.Ticker object.
    :Yields: yf.Ticker objects
    """
    all_data: dict[str, dict[str, str]] = featured.get_all_company_data_response()
    all_tickers: pd.Series = featured.get_full_list_of_tickers(all_data)
    all_cik: pd.Series = featured.get_full_list_cik(all_data)
    count = 0 #Temp

    #TODO Maybe make a data class
    for ticker_str in all_tickers:
        ticker: yf.Ticker = yf.Ticker(ticker_str)
        cik: int = all_cik.iloc[count]
        if count < 80: #Temp
            if filters.company_filter(ticker, by_industry=True, by_market_cap=True, by_profitability=True,
                                      by_public_age=True):
                count += 1
                yield ticker, cik, ticker_str

            else:
                count += 1
                continue

        else: # Temp
            return


def dev_price_data() -> pd.DataFrame:
    """
    Develops dataframe with price data for feature selection.
    :returns: dataframe with price data
    """
    df: pd.DataFrame = pd.DataFrame()

    for ticker, cik, ticker_str in dev_featured_companies():
        new_df: pd.DataFrame = price_data.build_single_ticker_price_df(ticker, open_p=True, close_p=True)
        df = pd.concat([df, new_df])

    return df

def dev_fundamental_data() -> pd.DataFrame:
    """
    Develops dataframe with fundamental data for feature selection.
    :returns: dataframe with fundamental data
    """
    df: pd.DataFrame = pd.DataFrame()
    for ticker, cik, ticker_str in dev_featured_companies():
        new_df: pd.DataFrame = build_single_ticker_fundamentals_df(cik, ticker_str, 10)
        df = pd.concat([df, new_df])

    return df


# def dev_full_data() -> pd.DataFrame:
#     """
#     TODO: NEEDS TO BE MODIFIED
#     """
#     df: pd.DataFrame = pd.DataFrame()
#     for ticker, cik in dev_featured_companies():
#         price_df: pd.DataFrame = dev_price_data(ticker)
#         # fundamental_df: pd.DataFrame = dev_fundamental_data(cik)
#         # df = pd.merge(price_df, fundamental_df, on=["Ticker", "Year", "Quarter"])
#
#     return df

def main():
    #TODO SAVE TO FILE NEEDS TO BE A METHOD
    # data: pd.DataFrame = dev_price_data()
    prices = dev_price_data()
    fundamentals = dev_fundamental_data()
    # TODO FIX MERGING
    # df = merge_df_columns([prices, fundamentals])
    # print(df)
    print(prices)
    print(fundamentals)
    read_write.write_to_csv(fundamentals, "example_fundamentals.csv")
    read_write.write_to_csv(prices, "example_prices.csv")

if __name__ == "__main__":
    main()

