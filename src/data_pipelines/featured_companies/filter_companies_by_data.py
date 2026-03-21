"""
File: filter_companies_by_data.py
Author: Drew Hill
Used for the purpose of filtering companies from a set based on to determine companies to use in the modeling pipeline.
"""
import pandas as pd
import yfinance as yf
import data_io.read_write_data as rw

import config

#TODO Might Not Need Might Need TO Filter After Collec
def filter_by_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    rows_to_drop = []
    for i in df.index:
        exchange = df.loc[i, "Exchange"]

        if exchange not in config.EXCHANGE:
            rows_to_drop.append(i)

    df = df.drop(index=rows_to_drop)

    return df

def filter_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param sectors:
    :param df:
    :return:
    """
    rows_to_drop = []
    for i in df.index:
        sector = df.loc[i, "Sector"]

        if sector not in config.SECTORS:
            rows_to_drop.append(i)

    df = df.drop(index=rows_to_drop)

    return df

def filter_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    rows_to_drop = []
    for i in df.index:
        industry = df.loc[i, "Industry"]

        if industry not in config.INDUSTRIES:
            rows_to_drop.append(i)

    df = df.drop(index=rows_to_drop)

    return df

def filter_by_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out companies by market cap from a list of companies.
    :return: A new dataframe of companies within the market cap range.
    """
    rows_to_drop = []

    for i in df.index:
        mcap = df.loc[i, "MarketCap"]
        if not (config.MIN_CAP <= mcap <= config.MAX_CAP):
            rows_to_drop.append(i)

    df = df.drop(index=rows_to_drop)

    return df

def filter_by_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the company has a level of profitability.
    :return: True if the company has a level of profitability, False otherwise.
    """
    rows_to_drop = []

    for i in df.index:
        margin = df.loc[i, "ProfitMargin"]

        if not margin >= config.MIN_PROFIT_MARGIN:
            rows_to_drop.append(i)

    df = df.drop(index=rows_to_drop)

    return df


def filter_by_public_age(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    rows_to_drop = []
    for i in df.index:
        age = df.loc[i, "TradingAge"]

        if age < config.MIN_PUBLIC_AGE:
            rows_to_drop.append(i)

    df = df.drop(index=rows_to_drop)

    return df

def filter_companies(
        from_file: str,
        by_exchange: bool | None = None,
        by_sector: bool | None = None,
        by_industry: bool | None = None,
        by_market_cap: bool | None = None,
        by_profitability: bool | None = None,
        by_public_age: bool | None = None
) -> pd.DataFrame:

    df = rw.read_from_csv(from_file)

    if by_exchange:
        df = filter_by_exchange(df)

    if by_sector:
        df = filter_by_sector(df)

    if by_industry:
        df = filter_by_industry(df)

    if by_market_cap:
        df = filter_by_market_cap(df)

    if by_profitability:
        df = filter_by_profitability(df)

    if by_public_age:
        df = filter_by_public_age(df)

    return df
