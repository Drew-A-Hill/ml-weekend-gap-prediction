"""
File: types_price_metrics.py
Author: Drew Hill
This file is used to hold argument typing schema
"""

from typing import TypedDict

class PriceMetrics(TypedDict, total=False):
    open_p: bool
    close_p: bool
    high_p: bool
    low_p: bool
    volume: bool
    dividends: bool
    stock_splits: bool
