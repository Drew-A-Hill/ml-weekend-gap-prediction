"""
File: company_retrieval.py
Author: Drew Hill
This file is used to find a list of companies that satisfy requirements needed to extract features for the model.
"""
from typing import Any

import pandas as pd

from src.data_pipelines.response import get_response

def get_all_company_data_response() -> dict[str, dict[str, str]]:
    """
    Retrieves and returns all registered with the SEC companies details, including, CIK, Ticker, Company Name.
    :return: Dictionary of companies details
    """
    url: str = "https://www.sec.gov/files/company_tickers.json"

    return get_response(url)

def get_full_list_of_tickers(data: dict[str, dict[str, str]]) -> pd.Series:
    """
    Parses out a panda series if all tickers related to companies registered with the SEC.
    :return: A panda series consisting of all companies registered with the SEC.
    """
    ticker_list: list[str] = []

    for company in data.values():
        ticker_list.append(company["ticker"].upper())

    return pd.Series(ticker_list)

def get_full_list_cik(data: dict[str, dict[str, str]]) ->pd.Series:
    """
    Parses out a panda series if all CIK's related to companies registered with the SEC.
    """
    cik_list: list[str] = []

    for company in data.values():
        cik_list.append(company["CIK"].upper())

    return pd.Series(cik_list)