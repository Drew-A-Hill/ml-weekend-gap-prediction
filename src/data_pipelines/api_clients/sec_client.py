"""

"""
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