"""

"""
import pandas as pd
import data_pipelines.api_data_ingestion.data_developer as collect
import data_io.read_write_data as rw


def dev_featured_companies() -> None:
    """

    :return:
    """
    companies: pd.DataFrame = collect.collect_companies_meta_data()
    rw.write_to_csv(companies, "company_meta_data.csv")