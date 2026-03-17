"""
File: indicator_calcs.py
Author: Drew Hill
This file is used for calculating fundamental indicators.
"""
import numpy as np
import pandas as pd
import data_pipelines.api_data_ingestion.indicator_calcs.intermediate_calcs as ic

def gross_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates gross margin as gross profit relative to revenue.
    :param df: The dataframe that contains all the fundamental data.
    :returns: The input dataframe with a gross_margin column added.
    """
    denom = df["Revenue"].replace(0, np.nan)
    df["gross_margin"] = df["GrossProfit"] / denom

    return df

def operating_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates operating margin as operating income relative to revenue.
    :param df: The dataframe that contains all the fundamental data.
    :returns: The input dataframe with an operating_margin column added.
    """
    denom = df["Revenue"].replace(0, np.nan)
    df["operating_margin"] = df["OperatingIncome"] / denom

    return df

def net_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates net margin as net income relative to revenue.
    :param df: The dataframe that contains all the fundamental data.
    :returns: The input dataframe with a net_margin column added.
    """
    denom = df["Revenue"].replace(0, np.nan)
    df["net_margin"] = df["NetIncome"] / denom

    return df

def debt_to_equity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates debt to equity ratio for each ticker.
    :param df: The dataframe that contains all the fundamental data.
    :returns: The input dataframe with a debt_to_equity_ratio column added.
    """
    denom = df["ShareholdersEquity"].replace(0, np.nan)
    df["debt_to_equity_ratio"] = df["TotalDebt"] / denom

    return df

def roa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates return on assets as net income relative to total assets.
    :param df: The dataframe that contains all the fundamental data.
    :returns: The input dataframe with a roa column added.
    """
    denom = df["TotalAssets"].replace(0, np.nan)
    df["roa"] = df["NetIncome"] / denom

    return df

def rev_growth_qoq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates quarter over quarter revenue growth for each ticker.
    :param df: The dataframe that contains all the fundamental data.
    :returns: The input dataframe with a rev_growth_qoq column added.
    """
    prev_rev = ic.prev_quarter_revenue(df).replace(0, np.nan)
    df["rev_growth_qoq"] = (df["Revenue"] / prev_rev) - 1

    return df