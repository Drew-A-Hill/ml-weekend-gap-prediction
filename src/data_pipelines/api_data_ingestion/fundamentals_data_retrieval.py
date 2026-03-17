"""
File: fundamentals_data_retrieval.py
Author: Drew Hill
This file is used for assembling all the needed fundamental data for feature selection.
"""
from typing import Any
import pandas as pd
import config
from config import FUNDAMENTAL_METRICS
from utils.pipline_helpers import get_response

def _filter_by_year(df: pd.DataFrame, num_years: int) -> pd.DataFrame | None:
    """
    Filter SEC data entries to include only desired years.
    :param df: Data frame of details from only desired years.
    :param num_years: Number of years to include.
    :Returns: Data for relevant years.
    """
    if df.empty:
        return df

    latest_fy: int = df["fy"].max()
    return df[df["fy"] >= latest_fy - (num_years - 1)]

def _filter_by_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SEC data entries to include only desired quarters.
    :param df: Data frame of details from only desired quarters.
    :Returns: Data for relevant quarters.
    """
    df = df[df["fp"].isin(config.QUARTERS)]

    return df

def _extract_metric(data: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    """
    Extract metric values from SEC JSON adds them to a list to be used for collection of fundamentals.
    :param data: SEC JSON data.
    :param tag: Metric tag.
    :Returns: List of metric values.
    """
    try:
        units = data["facts"]["us-gaap"][tag]["units"]

        # For USD metrics
        if "USD" in units:
            return units["USD"]

        # Returns for non USD metrics
        return next(iter(units.values()))

    except KeyError:
        return []

def _filter_by_forms_and_filings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters SEC data to only include the details from desired forms and filings.
    :param df: Data frame of details to be filtered.
    :Returns: Data for relevant forms and filings.
    """
    # Filters for only desired filings needed.
    if "form" in df.columns:
        df = df[df["form"].isin(["10-Q", "10-K"])]

    # Filers by filing date so latest values remain
    if "filed" in df.columns:
        df = df.sort_values("filed")

    return df

def _clean_metric(data: list[dict[str, Any]], years: int) -> list[dict[str, Any]]:
    """
    Cleans SEC metric data by removing duplicates and filtering filings.
    :param data: SEC data about the company being used.
    :param years: Number of years to include.
    :returns: Data for relevant years.
    """
    if not data:
        return []

    df = pd.DataFrame(data)

    df = _filter_by_quarter(df)
    df = _filter_by_forms_and_filings(df)

    # Remove duplicate fiscal periods
    df = df.drop_duplicates(subset=["fy", "fp"], keep="last")

    df = _filter_by_year(df, years)

    return df.to_dict("records")


def build_single_ticker_fundamentals_df(cik: int, ticker: str, years: int) -> pd.DataFrame:
    """
    Pull filters and returns data for selected fundamentals from SEC for a desired company.
    :param cik: CIK of company being used.
    :param ticker: Ticker symbol as a str for the company being used.
    :param years: Number of years to include.
    :returns: Data desired fundamental metrics for a desired company and time period.
    """
    padded_cik: str = str(cik).zfill(10)

    url: str = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json"
    data: dict[str, Any] = get_response(url)

    metric_frames: list[pd.DataFrame] = []

    for section in FUNDAMENTAL_METRICS.values():
        for metric_name, tags in section.items():

            if not isinstance(tags, list):
                tags = [tags]

            values: list[dict[str, Any]] = []

            for tag in tags:
                values = _extract_metric(data, tag)
                if values:
                    break

            values = _clean_metric(values, years)

            if not values:
                continue

            df = pd.DataFrame(values)

            df = df[["fy", "fp", "val"]]

            df.rename(
                columns={
                    "fy": "Year",
                    "fp": "Quarter",
                    "val": metric_name.upper(),
                },
                inplace=True,
            )

            metric_frames.append(df)

    if not metric_frames:
        return pd.DataFrame()

    # Merge all metrics together
    df_final = metric_frames[0]

    for df in metric_frames[1:]:
        df_final = df_final.merge(df, on=["Year", "Quarter"], how="outer")

    df_final.insert(0, "Ticker", ticker)

    df_final.sort_values(["Year", "Quarter"], inplace=True)

    latest_year = df_final["Year"].max()
    df_final = df_final[df_final["Year"] >= latest_year - (years - 1)]

    df_final.reset_index(drop=True, inplace=True)

    return df_final