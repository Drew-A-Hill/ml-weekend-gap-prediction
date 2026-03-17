"""
File: dev_price_data.py
Author: Drew Hill
This file is used to read data to a file and read from a file.
"""
import pandas as pd

from config import DATA_DIR

def write_to_csv(data: pd.DataFrame | pd.Series, file_name: str) -> None:
    """
    Writes a dataframe to a csv file.
    :param data: data to be written to csv file.
    :param file_name: file name to write to.
    """
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(DATA_DIR / file_name, index=False)

def read_from_csv(file_name: str) -> pd.DataFrame:
    """
    Reads from a csv file and return a dataframe.
    :param file_name: file name to read from.
    :return: a dataframe containing the data.
    """
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError

    return pd.read_csv(file_path)