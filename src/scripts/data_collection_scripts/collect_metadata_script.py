"""
This script collects all metadata for each ticker from the yf api
"""
from typing import Any

import pandas as pd
import data_io.read_write_data as rw
import data_pipelines.company_selection.meta_data_collection as collect
import data_pipelines.company_selection.registered_companies as registered
import utils.terminal_run_status as status

def collect_companies_meta_data() -> None:
    """

    :return:
    """
    featured_companies: list[dict[str, Any]] = []

    for ticker in status.ticker_iter_w_progress("Collecting Company Data", registered.get_all_tickers().index):
        featured_companies.append(collect.collect_filter_criteria_data(ticker))

    details: pd.DataFrame = pd.DataFrame(
        featured_companies,
        columns=["Ticker", "Exchange", "Sector", "Industry", "MarketCap", "ProfitMargin", "TradingAge"]
    )

    rw.write_to_csv(details, "company_meta_data.csv")