"""
File: filter_companies_from_df.py
Author: Drew Hill
Used for the purpose of filtering companies from a set based on to determine companies to use in the modeling pipeline.
"""
import pandas as pd
import data_io.read_write_data as rw
import config

def filter_by_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out companies that are not in designated exchange from a list of companies.
    :param df: DataFrame of companies to filter.
    :return: A new dataframe of companies within the exchanges designated..
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
    Filters out companies that are not in designated sector from a list of companies.
    :param df: DataFrame of companies to filter.
    :return: A new dataframe of companies within the sectors designated.
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
    Filters out companies that are not in designated industry from a list of companies.
    :param df: DataFrame of companies to filter.
    :return: A new dataframe of companies within the industries designated.
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
    Filters out companies that are not in designated market cap range from a list of companies.
    :param df: DataFrame of companies to filter.
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
    Filters out companies that are not of the designated profitability minimum from a list of companies.
    :param df: DataFrame of companies to filter.
    :return: A new dataframe of companies that have at least the minimum profitability.
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
    Filters out companies that are not of the minimum designated public age from a list of companies.
    :param df: DataFrame of companies to filter.
    :return: A new dataframe of companies that are at least the minimum public age.
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
    """
    Filters companies by designated filter and criteria.
    :param from_file: File that holds companies to filter.
    :param by_exchange: Flag to designate if companies should be filtered by exchange.
    :param by_sector: Flag to designate if companies should be filtered by sectors.
    :param by_industry: Flag to designate if companies should be filtered by industry.
    :param by_market_cap: Flag to designate if companies should be filtered by market_cap.
    :param by_profitability: Flag to designate if companies should be filtered by profitability.
    :param by_public_age: Flag to designate if companies should be filtered by public age.
    :return: A new dataframe of filtered companies.
    """
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
