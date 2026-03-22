"""

"""
import datetime
from typing import Any

import pandas as pd
import data_pipelines.api_clients.yahoo_client as yf_client


def get_exchange(company_info: dict[str, Any]) -> str:
    """

    :param company_info:
    :return:
    """
    return company_info.get("exchange")

def get_sector(company_info: dict[str, Any]) -> str:
    """

    :param company_info:
    :return:
    """
    return company_info.get("sector")

def get_industry(company_info: dict[str, Any]) -> str:
    """

    :param company_info:
    :return:
    """
    return company_info.get("industry")

def get_market_cap(company_info: dict[str, Any]) -> int:
    """

    :param company_info:
    :return:
    """
    return company_info.get("marketCap")

def get_profit_margin(company_info: dict[str, Any]) -> float:
    """

    :param company_info:
    :return:
    """
    return company_info.get("profitMargins")

def get_trading_age(ticker: str) -> int:
    """

    :return:
    """
    history: pd.DataFrame = yf_client.get_price_history(ticker, period="max")

    if history is None:
        return 0

    if history.empty:
        return 0

    first_trade_date: pd.Timestamp = history.index.min()

    return datetime.datetime.now().year - first_trade_date.year

def collect_filter_criteria_data(ticker: str) -> dict[str, Any]:
    """

    :param ticker:
    :return:
    """
    info: dict[str, Any] = yf_client.get_info(ticker)

    return {
        "Ticker": ticker,
        "Exchange": get_exchange(info),
        "Sector": get_sector(info),
        "Industry": get_industry(info),
        "MarketCap": get_market_cap(info),
        "ProfitMargin": get_profit_margin(info),
        "TradingAge": get_trading_age(ticker)
    }

