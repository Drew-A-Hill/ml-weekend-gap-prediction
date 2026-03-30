"""
File: price_aggregate.py
Author: Drew Hill
This file is used for calculating price aggregate technical indicators.
"""
import numpy as np
import pandas as pd
import data_pipelines.api_data_ingestion.indicator_calcs.intermediate_calcs as ic

# def weekly_return(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates weekly return as Friday close relative to Monday open.
#     :param df: The dataframe that contains all the price data.
#     :returns: The input dataframe with a weekly_return column added.
#     """
#     mon_open = ic.monday_open(df).replace(0, np.nan)
#     df["WeeklyReturn"] = (ic.friday_close(df) / mon_open) - 1
#
#     return df

def intra_week_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates intra-week volatility as the standard deviation of daily returns within each ticker-week.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with an intra_week_volatility column added.
    """
    df["IntraWeekVolatility"] = ic.daily_returns(df).groupby(
        [df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("std")

    return df

def weekly_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates weekly range as weekly high minus weekly low scaled by Monday open.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with a weekly_range column added.
    """
    mon_open = ic.monday_open(df).replace(0, np.nan)
    df["WeeklyRange"] = (ic.weekly_high(df) - ic.weekly_low(df)) / mon_open

    return df

def friday_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Friday position within the weekly trading range.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with a friday_position column added.
    """
    weekly_span = (ic.weekly_high(df) - ic.weekly_low(df)).replace(0, np.nan)
    df["FridayPosition"] = (ic.friday_close(df) - ic.weekly_low(df)) / weekly_span

    return df

def open_close_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Friday open-close spread as Friday close relative to Friday open.
    :param df: The dataframe that contains all the price data.
    :returns: The input dataframe with an open_close_spread column added.
    """
    fri_open = ic.friday_open(df).replace(0, np.nan)
    df["OpenCloseSpread"] = (ic.friday_close(df) / fri_open) - 1

    return df
