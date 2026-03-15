"""
File: dev_price_data.py
Author: Drew Hill
This file is used to read data to a file and read from a file.
"""

import pandas as pd

from config import DATA_DIR

def write_to_csv(data: pd.DataFrame, file_name: str) -> None:
    """

    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(DATA_DIR / file_name, index=False)

def read_from_csv(file_name: str) -> pd.DataFrame:
    """

    """
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError

    return pd.read_csv(file_path)