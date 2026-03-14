"""
File: featured_company_filters.py
Author: Drew Hill
Utilities for filtering companies based on industry, market capitalization, and profitability to determine the set of
companies used in the modeling pipeline.
"""
import datetime
from typing import Any

import pandas as pd
import yfinance as yf

import config

def filter_by_industry(company_info: dict[str, Any]) -> bool:
    """
    Checks that company is a part of the industry list.
    :param company_info: Info about company being evaluated.
    :return: True if the company is in the list of industries, False otherwise.
    """
    industry: str = company_info.get("industry")

    if industry and industry in config.INDUSTRIES:
        return True

    return False

def filter_by_market_cap(company_info: dict[str, Any]) -> bool | None:
    """
    Checks if the company is within the market cap range.
    :param company_info: Info about company being evaluated.
    :return: True if the company is within the market cap range, False otherwise.
    """
    m_cap: float = company_info.get("marketCap")

    if m_cap is None:
        return None

    if config.MIN_CAP <= m_cap <= config.MAX_CAP:
        return True

    return False

def filter_by_profitability(company_info: dict[str, Any]) -> bool:
    """
    Checks if the company has a level of profitability.
    :param company_info: Info about company being evaluated.
    :return: True if the company has a level of profitability, False otherwise.
    """
    margin: float = company_info.get("profitMargins")

    if margin and margin > config.MIN_PROFIT_MARGIN:
        return True

    return False

def filter_by_public_age(ticker: yf.Ticker, pub_age: int) -> bool:
    """
    Checks if the company has been publicly traded for at minimum the required number of years.
    :param ticker: Ticker of company being evaluated.
    :param pub_age: Number of years the company being evaluated has been publicly traded.
    """
    history: pd.DataFrame = ticker.history(period="max")
    first_trade_date: pd.Timestamp = history.index.min()

    public_age: int = datetime.datetime.now().year - first_trade_date.year

    return public_age >= pub_age

def company_filter(ticker: yf.Ticker,
                   by_industry: bool = False,
                   by_market_cap: bool = False,
                   by_profitability: bool = False,
                   by_public_age: bool = False
                   ) -> bool:
    """

    """
    company_info: dict[str, Any] = ticker.fast_info

    if by_market_cap:
        if filter_by_market_cap(company_info) is None:
            company_info = ticker.info

            return filter_by_market_cap(company_info)

        elif not filter_by_market_cap(company_info):
            return False

    company_info = ticker.info

    if by_industry:
        if not filter_by_industry(company_info):
            return False

    if by_profitability:
        if not filter_by_profitability(company_info):
            return False

    if by_public_age:
        if not filter_by_public_age(ticker, config.MIN_PUBLIC_AGE):
            return False

    return True









