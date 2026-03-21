"""
File: build_data_set_script.py
Author: Drew Hill
This file runs the date set builder.
"""
import pandas as pd

import data_io.read_write_data as rw
from data_pipelines.api_data_ingestion.data_developer import dev_price_and_fundamental_data_by_ticker

def dev_data_set() -> None:
    companies: pd.DataFrame = rw.read_from_csv("featured_companies.csv")
    print(dev_price_and_fundamental_data_by_ticker(companies))
