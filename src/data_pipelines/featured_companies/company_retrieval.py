"""
File: company_retrieval.py
Author: Drew Hill
This file is used to find a list of companies that satisfy requirements needed to extract features for the model.
"""
import pandas as pd

from utils.pipline_helpers import get_response
_ALL_COMPANIES_DF: pd.DataFrame | None = None

def _get_all_company_data_response() -> dict[str, dict[str, str]]:
    """
    Retrieves and returns all registered with the SEC companies details, including, CIK, Ticker, Company Name.
    :return: Dictionary of companies details
    """
    url: str = "https://www.sec.gov/files/company_tickers.json"

    return get_response(url)

def get_all_companies() -> pd.DataFrame:
    """
    Creates and populates a data frame containing ticker, cik, and title for all companies registered with the SEC.
    :return: A data frame consisting of all companies registered with the SEC.
    """
    global _ALL_COMPANIES_DF

    formated_data: list[tuple[str, str, str]] = []

    for company in _get_all_company_data_response().values():
        formated_data.append((company["ticker"], company["cik_str"], company["title"]))

    _ALL_COMPANIES_DF = pd.DataFrame(formated_data, columns=["ticker", "cik", "title"]).set_index("ticker")

    return _ALL_COMPANIES_DF

def get_all_tickers() -> pd.Series:
    """
    Parses out a panda series if all tickers related to companies registered with the SEC.
    :return: A panda series consisting of all companies registered with the SEC.
    """
    return get_all_companies().index.to_series()

def get_all_cik() ->pd.Series:
    """
    Gets the CIKs' for all companies registered with the SEC.
    :return: A panda series consisting of the CIKs'.
    """
    return pd.Series(get_all_companies()["cik"])

def get_cik(ticker: str) -> int:
    """
    Gets the CIK' for a company registered with the SEC.
    :param ticker: Company ticker.
    :return: The desired CIK'.
    """
    return get_all_companies()["cik"][ticker]