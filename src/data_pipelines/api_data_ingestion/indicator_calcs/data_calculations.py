"""
File: data_calculations.py
Author: Drew Hill
This file is used for calculating any values to be used for feature selection from provided metrics.
"""
import pandas as pd

def calc_daily_return(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and includes the daily return for the data provided.
    :param data: Dataframe that contains data used to calculate the daily return value.
    :return: New dataframe updated with the calculated daily return value.
    """
    data["daily_return"] = data["Close"].pct_change()

    return data

def calc_add_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and includes the volume weighted average price for the data provided.
    :param data: Dataframe that contains data used to calculate the volume weighted average price.
    :return: New dataframe updated with the calculated volume weighted average price.
    """
    tp: int = (data["High"] + data["Low"] + data["Close"]) / 3
    data["vwap"] = (tp * data["Volume"]).cumsum() / data["Volume"].cumsum()

    return data

def calc_add_weekly_avg(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and includes the weekly average price for the data provided.
    :param data: Dataframe that contains data used to calculate the weekly average price.
    :return: New dataframe updated with the calculated weekly average price.
    """
    data["weekly_avg_vol"] = data["Volume"].rolling(5).mean()

    return data
