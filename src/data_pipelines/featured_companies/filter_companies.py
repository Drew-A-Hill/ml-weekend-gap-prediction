"""
File: filter_companies.py
Author: Drew Hill
Used for the purpose of filtering companies from a set based on to determine companies to use in the modeling pipeline.
"""
import pandas as pd
import yfinance as yf

import config

#TODO Might Not Need Might Need TO Filter After Collec
def filter_by_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out companies by market cap from a list of companies.
    :return: A new dataframe of companies within the market cap range.
    """
    rows_to_drop = []

    for idx in df.index:
        mcap = df.loc[idx, "marketCap"]
        if not (config.MIN_CAP <= mcap <= config.MAX_CAP):
            rows_to_drop.append(idx)

    df = df.drop(index=rows_to_drop)

    return df

def filter_by_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the company has a level of profitability.
    :return: True if the company has a level of profitability, False otherwise.
    """
    pass

def filter_by_public_age(ticker: yf.Ticker, pub_age: int) -> bool:
    """
    Checks if the company has been publicly traded for at minimum the required number of years.
    :param ticker: Ticker of company being evaluated.
    :param pub_age: Number of years the company being evaluated has been publicly traded.
    :return: True if the company has been publicly traded for specified number of years, False otherwise.
    """
