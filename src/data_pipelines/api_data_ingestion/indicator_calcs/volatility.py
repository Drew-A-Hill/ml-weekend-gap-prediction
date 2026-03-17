"""
File: volatility.py
Author: Drew Hill
This file is used for calculating volatility technical indicators.
"""
import numpy as np
import pandas as pd

import data_pipelines.api_data_ingestion.indicator_calcs.intermediate_calcs as ic

def bollinger_band_width(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculates Bollinger Band width as the distance between upper and lower bands scaled by the rolling mean.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate Bollinger Band width.
    :returns: The input dataframe with a bollinger_band_width column added.
    """
    sma_n = ic.sma_20(df) if window == 20 else df.groupby("Ticker")["Close"].transform(
        lambda s: s.rolling(window).mean())

    ub = ic.upper_band(df, window)
    lb = ic.lower_band(df, window)
    denom = sma_n.replace(0, np.nan)
    df["bollinger_band_width"] = (ub - lb) / denom

    return df

def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates Average True Range for each ticker over the specified window.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the atr.
    :returns: The input dataframe with an atr column added.
    """
    df["atr"] = ic.tr(df).groupby(df["Ticker"]).transform(lambda s: s.rolling(window).mean())

    return df

def five_d_std_dev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 5-day rolling standard deviation of daily returns for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with a five_d_std_dev column added.
    """
    dr = ic.daily_returns(df)
    df["five_d_std_dev"] = dr.groupby(df["Ticker"]).transform(lambda s: s.rolling(5).std())

    return df