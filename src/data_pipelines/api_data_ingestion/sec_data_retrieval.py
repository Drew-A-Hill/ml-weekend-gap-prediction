"""
File name: sec_data_retrieval.py
Author: Drew Hill
Retrieves needed sec data through use of SEC API
"""
from time import sleep
from typing import Any

import pandas as pd
import requests
from requests_toolbelt import user_agent

from src import config
from src.data_pipelines.featured_companies.full_company_list import full_company_list

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
FS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
SLEEP_TIME: int = 1

def _response(url: str, cik: str) -> dict[str, Any]:
    """
    Retrieves SEC Edgar company submission for the specified CIK via API calls.
    :param cik: company pulled from index
    :return: JSON encoded company details
    """
    padded_cik: str = cik.zfill(10) # Pads CIK to fit API call format
    url: str = url.format(cik_padded=padded_cik)

    sleep(SLEEP_TIME) # API Call curtesy

    response = requests.get(url, headers=config.HEADER)
    response.raise_for_status()

    return response.json()

def filter_by_industry() -> list[str]:
    """
    Retrieves a filtered list of companies CIKs by industry.
    :return: A list of CIKs.
    """
    cik_by_industry: list[str] = list()

    for each_cik in full_company_list()["cik"]:
        response: dict[str, Any] = _response(each_cik)

        if response["sic"] in config.SIC:
            cik_by_industry.append(response["cik"])

    return cik_by_industry

def retrieve_all_financials(cik: str) -> dict[str, Any]:
    """
    Retrieves all financial data from SEC Edgar by company CIK.
    :return: All financial data
    """
    response: dict[str, Any] = _response(FS_URL, cik)

    return response["facts"]["us_gaap"]

def _retrieve_fundamentals(req_fundamentals: dict[str, Any]) -> dict[str, Any]:
    pass


def _retrieve_company_details(response: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieves SEC Edgar company details from json.
    :return: Company details if exists, None otherwise
    """
    # Holds the dictionary root node for the company location information
    location_root: dict[str, Any] = response.get("addresses", {}).get("business", {})

    # Checks for sic
    sic_str: str = response.get("sic")

    # Adds desired details to dictionary
    details: dict[str, Any] = {
        "sic": int(sic_str) if sic_str else None,
        "sic_description": response.get("sicDescription"),
        "tickers": response.get("tickers", [None])[0] if response.get("tickers") else None,
        "city": location_root.get("city"),
        "state_country": location_root.get("stateOrCountry"),
    }

    return details

def get_submission_details(user_agent: str, cik: int) -> dict[str, Any]:
    """
    Gets SEC Edgar company submission details for specified CIK from API calls.
    :param user_agent: Agent string to use for API calls
    :param cik: company ID pulled from index
    :return: JSON encoded company details
    """
    return _retrieve_details(_response(user_agent, cik))