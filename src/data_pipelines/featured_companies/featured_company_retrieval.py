"""
File: featured_company_retrieval.py
Author: Drew Hill
This file is used to find a list of companies that satisfy requirements needed to extract features for the model.
"""
from typing import Any

import pandas as pd
import yfinance as yf

import config
from src.data_pipelines.response import get_response

def full_list_of_tickers() -> pd.Series:
    """
    Retrieves and inserts all tickers registered with the SEC into a panda series.
    :return: A panda series consisting of all companies registered with the SEC.
    """
    url: str = "https://www.sec.gov/files/company_tickers.json"
    data: dict[str, dict[str, str]] = get_response(url)
    ticker_list: list[str] = []

    for company in data.values():
        ticker_list.append(company["ticker"].upper())

    return pd.Series(ticker_list)

def check_companies_by_industry(company_info: dict[str, Any], industry_list: list[str]) -> bool:
    """
    Checks that company is a part of the industry list.
    :param company_info: Info about company being evaluated.
    :param industry_list: List of industries to check company against.
    :return: True if the company is in the list of industries, False otherwise.
    """
    industry: str = company_info.get("industry")

    if industry and industry in industry_list:
        return True

    return False

def check_companies_by_market_cap(company_info: dict[str, Any]) -> bool:
    """
    Checks if the company is within the market cap range.
    :param company_info: Info about company being evaluated.
    :return: True if the company is within the market cap range, False otherwise.
    """
    m_cap: float = company_info.get("marketCap")

    if m_cap and config.MIN_CAP <= m_cap <= config.MAX_CAP:
        return True

    return False

def check_companies_by_profitability(company_info: dict[str, Any]) -> bool:
    """
    Checks if the company has a level of profitability.
    :param company_info: Info about company being evaluated.
    :return: True if the company has a level of profitability, False otherwise.
    """
    margin: float = company_info.get("profitMargins")

    if margin and margin > config.MIN_PROFIT_MARGIN:
        return True

    return False


