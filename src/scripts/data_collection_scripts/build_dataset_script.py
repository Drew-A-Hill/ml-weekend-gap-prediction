"""
File: build_dataset_script.py
Author: Drew Hill
This file is a script that is used to build a complete dataset.
"""
import pandas as pd
from data_io.read_write_data import read_from_csv, write_to_csv
from data_pipelines.api_data_ingestion.fundamentals_data_retrieval import get_fundamentals
from data_pipelines.api_data_ingestion.indicator_data_retrieval import add_indicators
from data_pipelines.api_data_ingestion.price_data_retrieval import get_price_data
from utils.terminal_run_status import ticker_iter_w_progress

def build_base_dataset_price_fundamentals() -> None:
    data = read_from_csv("./fetched_data/filtered_company_list.csv")
    df = pd.DataFrame()
    pdf = pd.DataFrame()

    for ticker in ticker_iter_w_progress("Building Dataset", data["Ticker"]):
        fd = get_fundamentals(ticker, "2013", "2026")
        df = pd.concat([df, fd])

        p_data = get_price_data(ticker, open_p=True, close_p=True, high_p=True, low_p=True, volume=True)
        pdf = pd.concat([pdf, p_data], ignore_index=True)

    df = pd.merge(pdf, df, on=["Ticker", "Year", "Quarter"], how="left")

    write_to_csv(df, "./datasets/base_dataset_price_fundamental.csv")

def include_indicators_script() -> None:
    """

    """
    df = read_from_csv("./datasets/base_dataset_price_fundamental.csv")
    data = add_indicators(df, add_all=True)
    write_to_csv(data, "./datasets/base_dataset_all.csv")

def main():
    build_base_dataset_price_fundamentals()
    include_indicators_script()

if __name__ == '__main__':
    main()
