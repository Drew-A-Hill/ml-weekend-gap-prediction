"""
File: featured_company_retrieval.py
Author: Drew Hill
This file is used to find a list of companies that satisfy requirements needed to extract features for the model.
"""
import pandas as pd
import yfinance as yf

from src.data_pipelines.response import get_response

def full_list_of_tickers() -> pd.Series:
    """
    Retrieves and inserts all tickers registered with the SEC into a panda series.
    :return: A panda series consisting of all companies registered with the SEC.
    """
    url: str = "https://www.sec.gov/files/company_tickers.json"
    return pd.Series(get_response(url))

def check_companies_by_sector(ticker: yf.Ticker, industry_list: list[str]) -> bool:
    """
    Checks that company is a part of the industry list.
    :param ticker: Ticker of the company being checked.
    :param industry_list: List of industries to check company against.
    :return: True if the company is in the list of industries, False otherwise.
    """
    if ticker.info["industry"] in industry_list:
        return True

    return False

def check_companies_by_market_cap(ticker: yf.Ticker, min_cap: int, max_cap: int) -> bool:
    """
    Checks if the company is within the market cap range.
    :param ticker: Ticker of the company being checked.
    :param min_cap: Minimum market cap allowable.
    :param max_cap: Maximum market cap allowable.
    :return: True if the company is within the market cap range, False otherwise.
    """
    if min_cap <= ticker.info["marketCap"] <= max_cap:
        return True

    return False

def check_companies_by_profitability(ticker: yf.Ticker, margin: float) -> bool:
    """
    Checks if the company has a level of profitability.
    :param margin: Profit margin used as the metric.
    :param ticker: Ticker of the company being checked.
    :return: True if the company has a level of profitability, False otherwise.
    """
    if ticker.info["profitMargins"] > margin:
        return True

    return False


