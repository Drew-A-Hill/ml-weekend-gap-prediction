"""
File: featured_company_filters.py
Author: Drew Hill
Used for the purpose of filtering companies based on industry, market capitalization, and profitability to determine the set of
companies used in the modeling pipeline.
"""
from typing import Any
import datetime
import pandas as pd
import yfinance as yf
from urllib3.exceptions import HTTPError
import config

def _filter_by_financial_market(company_info: dict[str, Any]) -> bool:
    """
    This is a helper filter that ensures that the companies being filtered are traded on the NYSE, NASDAQ, or American
    NYSE. This increases the probability that the company will have its market cap in the fast info.
    :param company_info: Info about company being evaluated.
    :return: True if the company is on one of the designated exchanges, False otherwise.
    """
    try:
        if company_info.get("exchange") in config.EXCHANGE:
            return True

        return False

    except HTTPError:
        return False

    except KeyError:
        return False

def filter_by_industry(company_info: dict[str, Any]) -> bool:
    """
    Checks that company is a part of the industry list.
    :param company_info: Info about company being evaluated.
    :return: True if the company is in the list of industries, False otherwise.
    """
    industry: str = company_info.get("industry")

    if industry in config.INDUSTRIES:
        return True

    return False

def filter_by_market_cap(company_info: dict[str, Any]) -> bool | None:
    """
    Checks if the company is within the market cap range.
    :param company_info: Info about company being evaluated.
    :return: True if the company is within the market cap range, False otherwise.
    """
    try:
        m_cap = company_info.get("marketCap")

        if m_cap is None:
            return None

        return config.MIN_CAP <= m_cap <= config.MAX_CAP

    except HTTPError:
        return None

def filter_by_profitability(company_info: dict[str, Any]) -> bool:
    """
    Checks if the company has a level of profitability.
    :param company_info: Info about company being evaluated.
    :return: True if the company has a level of profitability, False otherwise.
    """
    try:
        margin = company_info.get("profitMargins")

        if margin is not None and margin > config.MIN_PROFIT_MARGIN:
            return True

        return False

    except HTTPError:
        return False

def filter_by_public_age(ticker: yf.Ticker, pub_age: int) -> bool:
    """
    Checks if the company has been publicly traded for at minimum the required number of years.
    :param ticker: Ticker of company being evaluated.
    :param pub_age: Number of years the company being evaluated has been publicly traded.
    :return: True if the company has been publicly traded for specified number of years, False otherwise.
    """
    try:
        history: pd.DataFrame = ticker.history(period="max")

        if history.empty:
            return False

        first_trade_date: pd.Timestamp = history.index.min()
        public_age: int = datetime.datetime.now().year - first_trade_date.year

        return public_age >= pub_age

    except HTTPError:
        return False

def company_filter(ticker: str,
                   by_industry: bool = False,
                   by_market_cap: bool = False,
                   by_profitability: bool = False,
                   by_public_age: bool = False
                   ) -> bool:
    """
    Filters companies based for given filtering criteria.
    :param ticker: Ticker of company being evaluated.
    :param by_industry: Whether to filter by industry.
    :param by_market_cap: Whether to filter by market cap.
    :param by_profitability: Whether to filter by profitability.
    :param by_public_age: Whether to filter by public age.
    :return: True if the company meets the criteria, False otherwise.
    """
    ticker: yf.Ticker = yf.Ticker(ticker)
    company_info: dict[str, Any] = ticker.fast_info

    if not _filter_by_financial_market(company_info):
        return False

    if by_market_cap:
        market_cap_result: bool | None = filter_by_market_cap(company_info)

        if market_cap_result is None:
            company_info = ticker.info
            market_cap_result = filter_by_market_cap(company_info)

        if not market_cap_result:
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