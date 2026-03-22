"""
File: build_dataset_script.py
Author: Drew Hill
This file runs the date set builder.
"""
import pandas as pd

import data_io.read_write_data as rw
import data_pipelines.api_data_ingestion.data_developer as ddev
import utils.terminal_run_status as status
import data_pipelines.api_data_ingestion.price_data_retrieval as price_data
import data_pipelines.api_data_ingestion.fundamentals_data_retrieval as fun
from data_pipelines.company_selection.registered_companies import get_cik


def dev_data_set() -> None:
    companies: pd.DataFrame = rw.read_from_csv("filtered_company_list.csv")

    tickers: pd.Series = companies["Ticker"]

    df: pd.DataFrame = pd.DataFrame()

    for ticker in status.ticker_iter_w_progress("Building Dataset", tickers.index):

        price: pd.DataFrame = price_data.build_single_ticker_price_df(ticker)
        fundamental: pd.DataFrame = fun.build_single_ticker_fundamentals_df(get_cik(ticker), ticker, 10)

        data: pd.DataFrame = ddev.dev_dataset_by_ticker(price=price, fundamental=fundamental)

        df = df.concat([df, data])

    rw.write_to_csv(df, "dataset.csv")



