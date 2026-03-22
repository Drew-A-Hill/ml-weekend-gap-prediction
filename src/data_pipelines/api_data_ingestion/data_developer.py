"""
File: data_developer.py
Author: Drew Hill
This file is used to develop the data set to be used for the model.
"""
import pandas as pd

def dev_dataset_by_ticker(
        price: pd.DataFrame|None = None,
        fundamental: pd.DataFrame|None = None,
        indicators: pd.DataFrame|None = None
) -> pd.DataFrame:
    """
    This method is used to develop the data set to be used for the model. Data set can contain price, and or
    fundamental and or indicator data.
    :param price: Pandas DataFrame that contains all the prices data or none.
    :param fundamental: Pandas DataFrame that contains all the fundamental data or none.
    :param indicators: Pandas DataFrame that contains all the indicators data or none.
    :return: Pandas DataFrame that contains the assembled dataset.
    """
    df: pd.DataFrame = pd.DataFrame()

    if price is not None and not price.empty:
        df = price.copy()

    if fundamental is not None and not fundamental.empty:
        if df.empty:
            df = fundamental.copy()
        else:
            df = pd.merge(df, fundamental, on=["Ticker", "Year", "Quarter"], how="outer")

    if indicators is not None and not indicators.empty:
        if df.empty:
            df = indicators.copy()
        else:
            df = pd.merge(df, indicators, on=["Ticker", "Year", "Quarter"], how="outer")

    return df