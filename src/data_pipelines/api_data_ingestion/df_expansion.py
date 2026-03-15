"""
File: df_expansion.py
Author: Drew Hill
This file is for merging data frames or adding columns to an existing data frame.
"""
import pandas as pd

def merge_df_columns(data_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple dataframes into a single dataframe.
    :param data_frames: list of dataframes to merge.
    :return: merged dataframe.
    """
    df: pd.DataFrame = data_frames[0]

    for data in data_frames[1:]:
        df = pd.merge(data, df, on=["ticker", "year", "quarter"], how="inner")

    return df

def add_new_rows(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Adds new rows to a dataframe from a given dataframe.
    """
    pass