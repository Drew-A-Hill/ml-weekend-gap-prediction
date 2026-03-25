"""
File: trend.py
Author: Drew Hill
This file is used for calculating trend technical indicators.
"""
import numpy as np
import pandas as pd

import data_pipelines.api_data_ingestion.indicator_calcs.intermediate_calcs as ic

def close_v_ema50(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates close relative to the 50-period EMA for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with a close_v_ema50 column added.
    """
    ema50 = ic.ema_50(df).replace(0, np.nan)
    df["CloseVEma50"] = (df["Close"] / ema50) - 1

    return df

def close_v_sma20(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates close relative to the 20-period SMA for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with a close_v_sma20 column added.
    """
    sma20 = ic.sma_20(df).replace(0, np.nan)
    df["CloseVSma20"] = (df["Close"] / sma20) - 1

    return df

def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Average Directional Index for each ticker.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the adx.
    :returns: The input dataframe with an adx column added.
    """
    prev_high = df.groupby("Ticker")["High"].transform(lambda s: s.shift(1))
    prev_low = df.groupby("Ticker")["Low"].transform(lambda s: s.shift(1))

    up_move = df["High"] - prev_high
    down_move = prev_low - df["Low"]

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index,)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),index=df.index,)

    tr_vals = ic.tr(df)
    atr_vals = tr_vals.groupby(df["Ticker"]).transform(lambda s: s.rolling(window).mean()).replace(0, np.nan)

    plus_di = 100 * plus_dm.groupby(df["Ticker"]).transform(lambda s: s.rolling(window).mean()) / atr_vals
    minus_di = 100 * minus_dm.groupby(df["Ticker"]).transform(lambda s: s.rolling(window).mean()) / atr_vals

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / di_sum) * 100

    df["ADX"] = dx.groupby(df["Ticker"]).transform(lambda s: s.rolling(window).mean())

    return df