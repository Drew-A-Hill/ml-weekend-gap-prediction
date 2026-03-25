"""

"""
from datetime import datetime
from typing import Any

from data_pipelines.api_clients.sec_client import get_response

def pad_cik(cik: int) -> str:
    """

    """
    return str(cik).zfill(10)

def fetch_facts(cik: str) -> dict:
    """

    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return get_response(url)

def parse_date(date_str: str) -> datetime:
    """

    """
    """Accept 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'."""
    date_str = date_str.strip()
    if len(date_str) == 4:
        return datetime.strptime(date_str, "%Y")
    if len(date_str) == 7:
        return datetime.strptime(date_str, "%Y-%m")
    return datetime.strptime(date_str, "%Y-%m-%d")

def duration_days(start_str: str, end_str: str) -> int:
    """

    """
    s = datetime.strptime(start_str, "%Y-%m-%d")
    e = datetime.strptime(end_str,   "%Y-%m-%d")
    return (e - s).days

def calc_gross_profit(row: dict[str, Any]) -> int:
    """

    """
    rev: int = row.get("revenues")
    cost: int = row.get("cost_of_revenues")

    return (rev - cost) if (rev is not None and cost is not None) else None