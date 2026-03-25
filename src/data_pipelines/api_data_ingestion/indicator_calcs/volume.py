"""
File: volatility.py
Author: Drew Hill
This file is used for calculating volume technical indicators.
"""
import numpy as np
import pandas as pd

import data_pipelines.api_data_ingestion.indicator_calcs.intermediate_calcs as ic

def obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates On-Balance Volume for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with an obv column added.
    """
    price_diff = df.groupby("Ticker")["Close"].transform(lambda s: s.diff())
    signed_volume = pd.Series(
        np.where(price_diff > 0, df["Volume"], np.where(price_diff < 0, -df["Volume"], 0)),
        index=df.index,
    )
    df["obv"] = signed_volume.groupby(df["Ticker"]).cumsum()

    return df

def mfi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates Money Flow Index for each ticker.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the mfi.
    :returns: The input dataframe with an mfi column added.
    """
    tp = ic.typical_price(df)
    rmf = ic.raw_money_flow(df)
    tp_change = tp.groupby(df["Ticker"]).transform(lambda s: s.diff())

    positive_flow = pd.Series(np.where(tp_change > 0, rmf, 0.0), index=df.index)
    negative_flow = pd.Series(np.where(tp_change < 0, rmf, 0.0), index=df.index)

    pos_sum = positive_flow.groupby(df["Ticker"]).transform(lambda s: s.rolling(window).sum())
    neg_sum = negative_flow.groupby(df["Ticker"]).transform(
        lambda s: s.rolling(window).sum()).replace(0, np.nan)

    money_ratio = pos_sum / neg_sum
    df["mfi"] = 100 - (100 / (1 + money_ratio))

    return df

def volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Friday volume relative to the average Monday through Thursday volume within each ticker-week.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with a volume_ratio column added.
    """
    mon_thu_vol = df["Volume"].where(df["Date"].dt.weekday < 4)
    mon_thu_avg = mon_thu_vol.groupby(
        [df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("mean").replace(0, np.nan)

    fri_vol = ic.friday_volume(df)
    df["volume_ratio"] = fri_vol / mon_thu_avg

    return df