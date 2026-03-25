"""
File: fundamentals_data_retrieval.py
Author: Drew Hill

Fetches fundamental financial data for a single company from SEC EDGAR and returns a pandas DataFrame, one row per
filing period.

Usage
-----
    from get_fundamentals import get_fundamentals

    df = get_fundamentals(ticker="NOW", start="2019", end="2024")
"""
from typing import Any
import pandas as pd
from datetime import datetime
from collections import defaultdict
import config as config
import data_pipelines.api_data_ingestion.fundamental_helpers as helpers
from data_pipelines.company_selection.registered_companies import get_cik


# ---------------------------------------------------------------------------
# is metric extraction  ->  {(end_date, fp): value}
# ---------------------------------------------------------------------------

def _extract_is(
    us_gaap: dict,
    tags:    list,
    start:   datetime,
    end:     datetime,
) -> dict:
    """
    Extracts an income statement metric across all tags, merging results so older tags fill in years where newer tags
    have no data.

    Each tag is scanned independently and its results are merged into a single
    dict keyed by (end_date, fp). When the same key appears in multiple tags
    the first (highest-priority) tag wins, so tag order still matters.

    Duplicate / YTD handling
    ------------------------
    EDGAR 10-Qs file both a single-quarter value (~91 days) and a YTD value
    (~273 days) with the same fp. We prefer single-quarter entries; when
    duplicates survive we take MAX per caller spec.

    Returns {} if no tag produced any usable data at all.
    """
    unit_key = "USD"
    merged: dict[tuple[str, str], float] = {}

    for tag in tags:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            continue

        filings = tag_data.get("units", {}).get(unit_key, [])
        if not filings:
            for v in tag_data.get("units", {}).values():
                if v:
                    filings = v
                    break
        if not filings:
            continue

        buckets = defaultdict(list)   # (end_d, fp) -> [(val, filed, is_single_q)]

        for entry in filings:
            fp: str = entry.get("fp", "")
            form: str = entry.get("form", "")
            val: float = entry.get("val")
            end_d: str = entry.get("end", "")
            start_d: str = entry.get("start", "")
            filed: str = entry.get("filed", "")

            if val is None or not end_d or fp not in config.VALID_FP:
                continue

            if fp == "FY" and form not in config.ANNUAL_FORMS:
                continue

            if fp in {"Q1","Q2","Q3","Q4"} and form not in config.QUARTERLY_FORMS:
                continue

            try:
                if helpers.parse_date(end_d) < start or helpers.parse_date(end_d) > end:
                    continue
            except ValueError:
                continue

            is_single_q = False
            if fp in {"Q1","Q2","Q3","Q4"} and start_d:
                try:
                    days = helpers.duration_days(start_d, end_d)
                    is_single_q = config.SINGLE_Q_MIN <= days <= config.SINGLE_Q_MAX
                except ValueError:
                    pass

            # Reject FY entries that are actually single-quarter segment
            # breakdowns (e.g. PAYX files sub-line revenue in 10-K with
            # fp=FY but duration ~90 days). True annual periods span >=340 days.
            if fp == "FY" and start_d:
                try:
                    if helpers.duration_days(start_d, end_d) < 340:
                        continue
                except ValueError:
                    pass

            buckets[(end_d, fp)].append((val, filed, is_single_q))

        # Resolve each bucket to a single value for this tag
        tag_result: dict[tuple[str, str], float] = {}
        for (end_d, fp), candidates in buckets.items():
            if fp in {"Q1","Q2","Q3","Q4"}:
                singles: list = [c for c in candidates if c[2]]
                pool: list = singles if singles else candidates
                tag_result[(end_d, fp)] = max(c[0] for c in pool)

            else:
                best = max(candidates, key=lambda x: x[1])
                tag_result[(end_d, fp)] = best[0]

        # Merge into combined result — first tag (highest priority) wins
        for key, val in tag_result.items():
            if key not in merged:
                merged[key] = val

    return merged


# ---------------------------------------------------------------------------
# BS metric extraction  ->  {end_date_str: value}
# ---------------------------------------------------------------------------

def _extract_bs(us_gaap: dict, tags: list, metric: str, start: datetime, end: datetime) -> dict:
    """
    Extract a balance sheet metric.
    Returns {end_date_str: value} — keyed solely by end date.

    Balance sheet values are point-in-time snapshots uniquely identified
    by their date regardless of which filing reported them. We keep the
    earliest-filed value for each date to avoid stock-split restatements
    from later 10-Ks overwriting the contemporaneous figure.
    """
    unit_key = "shares" if metric == "shares" else "USD"

    for tag in tags:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            continue

        filings = tag_data.get("units", {}).get(unit_key, [])
        if not filings:
            for v in tag_data.get("units", {}).values():
                if v:
                    filings = v
                    break
        if not filings:
            continue

        # end_date -> (val, filed)
        result = {}

        for entry in filings:
            form  = entry.get("form", "")
            val   = entry.get("val")
            end_d = entry.get("end", "")
            filed = entry.get("filed", "")

            if val is None or not end_d:
                continue
            if form not in config.ANNUAL_FORMS and form not in config.QUARTERLY_FORMS:
                continue

            try:
                end_dt = helpers.parse_date(end_d)
            except ValueError:
                continue
            if end_dt < start or end_dt > end:
                continue

            existing = result.get(end_d)
            # Earliest filing wins (contemporaneous value)
            if existing is None or filed < existing[1]:
                result[end_d] = (val, filed)

        if result:
            return {k: v for k, (v, _) in result.items()}

    return {}


# ---------------------------------------------------------------------------
# Q4 derivation
# ---------------------------------------------------------------------------

def _derive_q4(flow: dict) -> None:
    """
    Q4 is not filed in EDGAR — derive as Q4 = FY - Q1 - Q2 - Q3.
    The FY entry is removed after derivation; if Q1/Q2/Q3 are unavailable
    the FY value is promoted directly to Q4.

    Works for any fiscal year end month (Dec, May, Sep, Jan, etc.) by
    matching quarters to their FY via date ranges rather than calendar year:
    a quarter belongs to an FY if its end_date falls within
    (previous_FY_end, current_FY_end].
    """
    for metric in config.IS_METRICS:
        period_map = flow.get(metric)
        if not period_map:
            continue

        # Collect and sort all FY end dates
        fy_dates = sorted(
            [end_d for (end_d, fp) in period_map if fp == "FY"]
        )
        if not fy_dates:
            continue

        # For each FY, find Q1/Q2/Q3 whose end_date falls in
        # (prev_fy_end, fy_end] — works for any fiscal year calendar
        prev_fy = None
        for fy_end_d in fy_dates:
            fy_val = period_map.get((fy_end_d, "FY"))
            if fy_val is None:
                prev_fy = fy_end_d
                continue

            # Q1/Q2/Q3 end_dates that belong to this fiscal year
            qv = {}
            for (end_d, fp) in list(period_map.keys()):
                if fp not in ("Q1", "Q2", "Q3"):
                    continue
                # Must fall after previous FY end (if any) and on/before this FY end
                if prev_fy is not None and end_d <= prev_fy:
                    continue
                if end_d > fy_end_d:
                    continue
                qv[fp] = period_map[(end_d, fp)]

            q1, q2, q3 = qv.get("Q1"), qv.get("Q2"), qv.get("Q3")

            if q1 is not None and q2 is not None and q3 is not None:
                period_map[(fy_end_d, "Q4")] = fy_val - q1 - q2 - q3
            else:
                # Can't derive — promote FY as Q4
                period_map[(fy_end_d, "Q4")] = fy_val

            # Always remove the FY entry — Q4 represents year-end
            del period_map[(fy_end_d, "FY")]
            prev_fy = fy_end_d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rename_cols(row: dict) -> dict[str, Any]:
    """

    """
    row["Revenues"] = row.pop("revenues", None)
    row["CostOfRevenues"] = row.pop("cost_of_revenues", None)
    row["NetIncome"] = row.pop("net_income", None)
    row["Assets"] = row.pop("assets", None)
    row["Liabilities"] = row.pop("liabilities", None)
    row["Equity"] = row.pop("equity", None)
    row["Shares"] = row.pop("shares", None)

    return row

def get_fundamental_cols() -> list[str]:
    """

    """
    return [
        "Revenues", "CostOfRevenues", "GrossProfit",
        "NetIncome", "Assets", "Liabilities", "Equity", "Shares",
    ]

def set_col_order(fundamental_cols: list[str]) -> list[str]:
    """

    """


    return ["Ticker", "Year", "Quarter"] + fundamental_cols

def get_fundamentals(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch fundamental financial data for a single company from SEC EDGAR.

    :param ticker: exchange ticker symbol.
    :param start : start of period, including formats: "YYYY", "YYYY-MM", or "YYYY-MM-DD"
    :param end: end of period, including formats: "YYYY", "YYYY-MM", or "YYYY-MM-DD"

    :returns: pandas dataframe columns: ticker, cik, end_date, year, quarter, period, revenues, cost_of_revenues,
            gross_profit, profit_margin, net_income, assets, liabilities, equity, shares

    One row per (end_date, period). Sorted by end_date ascending.
    """
    cik: int = get_cik(ticker)
    cik_padded: str = helpers.pad_cik(cik)
    start_dt: datetime = helpers.parse_date(start)
    end_dt: datetime = helpers.parse_date(end)

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    facts: dict[str, Any] = helpers.fetch_facts(cik_padded)
    us_gaap: dict[str, Any] = facts.get("facts", {}).get("us-gaap", {})

    # --- Extract income_statement metrics: {(end_date, fp): value} ---
    income_statement: dict[str, Any] = {}
    for metric in config.IS_METRICS:
        income_statement[metric] = _extract_is(
            us_gaap, config.FUNDAMENTAL_METRICS[metric], start_dt, end_dt
        )

    # --- Derive Q4 ---
    _derive_q4(income_statement)

    # --- Extract bs metrics: {end_date: value} ---
    bs: dict[str, Any] = {}
    for metric in config.BS_METRICS:
        bs[metric] = _extract_bs(
            us_gaap, config.FUNDAMENTAL_METRICS[metric], metric, start_dt, end_dt
        )

    # --- Build rows from income_statement metric keys only ---
    # Collect all (end_date, fp) keys present in ANY income_statement metric
    all_keys: set = set()
    for period_map in income_statement.values():
        all_keys.update(period_map.keys())

    if not all_keys:
        return pd.DataFrame(columns=[
            "Ticker", "Year", "Quarter",
            "Revenues", "CostOfRevenues", "GrossProfit",
            "NetIncome", "Assets", "Liabilities", "Equity", "Shares",
        ])

    QUARTER_MAP = {"Q1": "Q1", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4"}

    rows = []
    for end_date, period in sorted(all_keys):
        # FY entries are removed by _derive_q4; skip any that remain
        # (e.g. if Q1/Q2/Q3 were unavailable and FY couldn't be consumed)
        if period == "FY":
            continue

        end_dt_row: datetime = datetime.strptime(end_date, "%Y-%m-%d")

        row: dict[str, Any] = {
            "Ticker": ticker,
            "Year": end_dt_row.year,
            "Quarter": QUARTER_MAP.get(period),
        }

        for metric in config.IS_METRICS:
            row[metric] = income_statement[metric].get((end_date, period))

        for metric in config.BS_METRICS:
            row[metric] = bs[metric].get(end_date)

        row["GrossProfit"] = helpers.calc_gross_profit(row)

        row = rename_cols(row)
        rows.append(row)

    df = pd.DataFrame(rows, columns=set_col_order(get_fundamental_cols()))
    df = df.sort_values(["Year", "Quarter"], key=lambda col: col.map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}) if col.name == "Quarter" else col).reset_index(drop=True)

    # Fill missing quarters within each (Ticker, Year) group:
    #   - forward-fill: Q2 gets Q1 if Q2 is NaN, Q3 gets Q2, Q4 gets Q3
    #   - backward-fill: Q1 gets Q2 if Q1 is NaN, Q2 gets Q3, etc.
    df[get_fundamental_cols()] = (
        df.groupby(["Ticker", "Year"], sort=False)[get_fundamental_cols()]
          .transform(lambda g: g.ffill().bfill())
    )

    return df