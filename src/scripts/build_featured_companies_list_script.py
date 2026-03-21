"""
File: build_featured_companies_list.py
Author: Drew Hill
This script is used to assemble the companies used for the model.
"""
import data_io.read_write_data as rw
from data_pipelines.featured_companies import filter_companies_by_data


def build_featured_companies_list() -> None:

    details = filter_companies_by_data.filter_companies(
        "company_meta_data.csv",
        by_exchange=True,
        by_market_cap=True,
        by_public_age=True,
        by_profitability=True,
        by_sector=True
    )

    rw.write_to_csv(details["Ticker"], "featured_companies_list.csv")