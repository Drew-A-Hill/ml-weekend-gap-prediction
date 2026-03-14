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

#TODO MAYBE MOVE THIS TO FILTER AND MAKE THE DATA AN INPUT
def dev_featured_companies() -> Generator[yf.Ticker] | None:
    """
    Develops the companies to be used in the model based on the criteria and yields each yf.Ticker object.
    :Yields: yf.Ticker objects
    """
    all_data: dict[str, dict[str, str]] = featured.get_all_company_data_response()
    all_tickers: pd.Series = featured.get_full_list_of_tickers(all_data)
    count = 0 #Temp

    #TODO Maybe make a data class
    for ticker_str in all_tickers:
        ticker: yf.Ticker = yf.Ticker(ticker_str)
        if count < 80: #Temp
            if filters.company_filter(ticker, by_industry=True, by_market_cap=True, by_profitability=True,
                                      by_public_age=True):
                count += 1
                yield ticker

            else:
                count += 1
                continue

        else: # Temp
            return


def dev_price_data() -> pd.DataFrame:
    """
    Develops dataframe with data for feature selection.
    :return: pd.DataFrame
    """
    df: pd.DataFrame = pd.DataFrame()

    for ticker in dev_featured_companies():
        new_df: pd.DataFrame = price_data.build_custom_single_price_df(ticker, open_p=True, close_p=True)
        df = pd.concat([df, new_df])

    return df

def dev_fundamental_data() -> pd.DataFrame:
    """

    """
    pass

def dev_full_data() -> pd.DataFrame:
    """
    TODO: NEEDS TO BE MODIFIED
    """
    df: pd.DataFrame = pd.DataFrame()
    for ticker, cik in dev_featured_companies():
        price_df: pd.DataFrame = dev_price_data(ticker)
        fundamental_df: pd.DataFrame = dev_fundamental_data(cik)
        df = pd.merge(price_df, fundamental_df, on=["Ticker", "Year", "Quarter"])

    return df

def main():
    #TODO SAVE TO FILE NEEDS TO BE A METHOD
    data: pd.DataFrame = dev_price_data()

    data_dir = Path(__file__).resolve().parents[1] / "structured_csv_data_files"
    data_dir.mkdir(parents=True, exist_ok=True)

    data.to_csv(data_dir / "file.csv", index=False)

    print(data.head(150))

if __name__ == "__main__":
    main()

