"""
File: momentum.py
Author: Drew Hill
This file is used for calculating momentum technical indicators.
"""
import numpy as np
import pandas as pd

import data_pipelines.api_data_ingestion.indicator_calcs.intermediate_calcs as ic

def rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Adds the Relative Strength Index for each ticker using rolling average gains and losses.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the rsi.
    :returns: The input dataframe with a rsi column added.
    """
    rs = ic.avg_gain(df, window) / ic.avg_loss(df, window).replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df

def macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates MACD as the 12-period EMA minus the 26-period EMA for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with a macd column added.
    """
    df["macd"] = ic.ema_12(df) - ic.ema_26(df)

    return df

def roc(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Calculates rate of change over the specified window for each ticker.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the rate of change.
    :returns: The input dataframe with a roc column added.
    """
    prev_close_n = df.groupby("Ticker")["Close"].transform(lambda s: s.shift(window)).replace(0, np.nan)
    df["roc"] = (df["Close"] / prev_close_n) - 1

    return df

def stoch_perc_k(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates stochastic percent K for each ticker.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate stochastic percent K.
    :returns: The input dataframe with a stoch_perc_k column added.
    """
    rolling_high = df.groupby("Ticker")["High"].transform(lambda s: s.rolling(window).max())
    rolling_low = df.groupby("Ticker")["Low"].transform(lambda s: s.rolling(window).min())
    denom = (rolling_high - rolling_low).replace(0, np.nan)
    df["stoch_perc_k"] = (df["Close"] - rolling_low) / denom

    return df

