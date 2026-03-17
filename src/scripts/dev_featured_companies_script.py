"""
File: dev_featured_companies_script.py
Author: Drew Hill
This script is used to assemble the companies used for the model.
"""
import pandas as pd

import data_io.read_write_data as rw
import data_pipelines.api_data_ingestion.data_developer as x

def dev_featured_companies() -> None:
    companies: pd.Series = x.dev_featured_companies(
        by_industry=True,
        by_public_age=True
    )

    rw.write_to_csv(companies, "featured_companies.csv")