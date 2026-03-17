"""
File: pip_helpers.py
Author: Drew Hill
This file is used for the helpers related to the data ingestion pipeline.
"""
from typing import Any
import requests
import config

def get_response(url: str) -> dict[str, dict[str, str]]:
    """
    Send an HTTP GET request to the provided URL and returns a parsed JSON.
    :param url: url used to retrieve desired data.
    :return: returns a parsed JSON.
    """
    try:
        return requests.get(url=url, headers=config.HEADER).json()

    except requests.exceptions.RequestException as e:
        raise ValueError(e)

def get_list_of_req_metrics(sig: dict[str, Any]) -> list[str]:
    """
    Helper method for that retrieves the list of required metrics for a ticker as marked in the parameters of the
    method signature.
    :param sig: Parameters and input as a dictionary.
    :return: list that contains the required metrics for a ticker.
    """
    include_list: list[str] = []

    for param, input_bool in sig.items():
        if isinstance(input_bool, bool):
            if input_bool:
                include_list.append(param)

    return include_list

def pad_cik(cik: str) -> str:
    """
    Ensures that the provided CIK has a minimum of 10 characters by adding 0's to front of cik.
    :param cik: unpadded CIK.
    :returns: padded CIK
    """
    return cik.zfill(10)