"""
File name: build_filtered_company_list_script.py
Author: Drew Hill
This is a script used to build a filtered list of companies.
"""
import pandas as pd

import data_io.read_write_data as rw
import data_pipelines.company_selection.filter_companies_from_df as filters

def build_filtered_list() -> None:
    filtered: pd.DataFrame = filters.filter_companies(
        "company_meta_data.csv",
        by_exchange=True,
        by_industry=True,
        by_market_cap=True,
        by_profitability=True,
        by_public_age=True
    )

    rw.write_to_csv(filtered, "filtered_company_list.csv")
