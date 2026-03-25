"""
SEC EDGAR Fundamental Data Retriever
Fetches financial metrics for a company by CIK, returning a pandas DataFrame
with ticker, cik, fiscal_year, and period (Q1-Q4 / FY) as merge keys.
"""

import requests
import pandas as pd
from typing import Optional
from collections import defaultdict

import config

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
# FUNDAMENTAL_METRICS = {
#     "income_statement": {
#         "total_revenue": [
#             # Standard post-ASC-606 tag — filed undimensioned by most companies;
#             # for companies like NOW that only file it dimensioned, the extractor
#             # will sum the segment members automatically.
#             "RevenueFromContractWithCustomerExcludingAssessedTax",
#             # Common alternative for companies that include assessed tax in revenue
#             "RevenueFromContractWithCustomerIncludingAssessedTax",
#             # Pre-ASC-606 / older filers
#             "Revenues",
#             "SalesRevenueNet",
#             "SalesRevenueGoodsNet",
#             "SalesRevenueServicesNet",
#         ],
#         "cost_of_revenue": [
#             # Preferred: single undimensioned total cost line (most common)
#             "CostOfRevenue",
#             # Used by companies (e.g. NOW) that report a combined cost-of-goods-
#             # and-services line instead of separate CostOfGoodsSold / CostOfServices
#             "CostOfGoodsAndServicesSold",
#             # Older / goods-only filers
#             "CostOfGoodsSold",
#             # Services-only filers
#             "CostOfServices",
#             # Some filers use CostOfSales as the top-level line
#             "CostOfSales",
#         ],
#         "gross_profit": [
#             "GrossProfit",
#         ],
#         "operating_income": [
#             "OperatingIncomeLoss",
#         ],
#         "net_income": [
#             "NetIncomeLoss",
#             # Partnerships / consolidated entities
#             "ProfitLoss",
#             # Some filers use the attributable-to-parent variant
#             "NetIncomeLossAvailableToCommonStockholdersBasic",
#         ],
#         "eps_basic": [
#             "EarningsPerShareBasic",
#             # Used when basic and diluted are identical
#             "EarningsPerShareBasicAndDiluted",
#         ],
#         "eps_diluted": [
#             "EarningsPerShareDiluted",
#             "EarningsPerShareBasicAndDiluted",
#         ],
#     },
#     "balance_sheet": {
#         "cash_and_cash_equivalents": [
#             "CashAndCashEquivalentsAtCarryingValue",
#             # Some filers roll cash + restricted cash together
#             "CashCashEquivalentsAndShortTermInvestments",
#         ],
#         "total_assets": [
#             "Assets",
#         ],
#         "current_assets": [
#             "AssetsCurrent",
#         ],
#         "total_liabilities": [
#             "Liabilities",
#             "LiabilitiesAndStockholdersEquity",  # rare fallback
#         ],
#         "current_liabilities": [
#             "LiabilitiesCurrent",
#         ],
#         "stockholders_equity": [
#             "StockholdersEquity",
#             # Includes non-controlling interest; common for consolidated groups
#             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
#         ],
#         "shares_outstanding": [
#             "CommonStockSharesOutstanding",
#             # Some filers report on a class-by-class basis; this is the fallback
#             "CommonStockSharesIssued",
#         ],
#     },
#     "cash_flow": {
#         "operating_cash_flow": [
#             "NetCashProvidedByUsedInOperatingActivities",
#         ],
#         "capital_expenditures": [
#             "PaymentsToAcquirePropertyPlantAndEquipment",
#             # Accrual-basis capex (non-cash); less common but present in some filers
#             "CapitalExpendituresIncurredButNotYetPaid",
#             # Some filers use the combined PP&E + intangibles line
#             "PaymentsToAcquireProductiveAssets",
#         ],
#     },
# }

# Metrics that are point-in-time (balance sheet) vs period (income/cash flow).
# These must match the metric KEY names used in config / FUNDAMENTAL_METRICS.
INSTANTANEOUS_METRICS = {
    # module-level default names
    "cash_and_cash_equivalents",
    "total_assets",
    "current_assets",
    "total_liabilities",
    "current_liabilities",
    "stockholders_equity",
    "shares_outstanding",
    # config-level names (bare column names)
    "Assets",
    "Liabilities",
    # NOTE: "Shares" is intentionally excluded — weighted-average share counts
    # (used as fallback for companies like WDAY) are duration/flow metrics,
    # not instant.  CommonStockSharesOutstanding IS instant, but since the tag
    # list now includes weighted-avg fallbacks we treat Shares as flow so the
    # correct end-date year is used for all candidates.
}

HEADERS = {"User-Agent": "fundamental-data-fetcher admin@example.com"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_cik(cik: str | int) -> str:
    """Zero-pad CIK to 10 digits as required by EDGAR."""
    return str(int(cik)).zfill(10)


def _fetch_company_facts(cik: str) -> dict:
    """Download the full company-facts JSON from EDGAR."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _is_dimensioned(entry: dict) -> bool:
    """
    Return True if an EDGAR filing entry carries a segment/dimension member,
    meaning it represents a sub-total rather than the consolidated figure.

    EDGAR's companyfacts JSON signals a dimension via an "accn" that maps to
    a context with explicit members.  In practice the raw entry carries a
    "segment" key when dimensioned (visible in some older dumps), or more
    reliably the same (accn, start, end) triple appears multiple times — once
    per member — whereas the undimensioned total appears exactly once.

    The most reliable signal available without fetching the full instance
    document is the presence of duplicate (accn, end, fp) triples in the
    filings list.  We handle this in _extract_period_values by grouping and
    comparing counts, but for a single-entry check we rely on the "segment"
    key that EDGAR includes for explicitly-dimensioned facts.
    """
    return "segment" in entry


def _extract_period_values(
    facts: dict,
    tag_candidates: list[str],
    metric_name: str,
    start_fy: int,
    end_fy: int,
    is_instant: bool,
) -> dict[tuple[int, str], float]:
    """
    Search tag_candidates in order; return the first tag that has usable data.

    Returns a dict keyed by (fiscal_year, fp) where fp is one of:
      "FY", "Q1", "Q2", "Q3", "Q4"

    KEY DESIGN: use `end` date year as fiscal_year, NOT the `fy` field
    ---------------------------------------------------------------
    EDGAR's `fy` field on each entry is the fiscal year of the *filing* that
    contains it, not the fiscal year of the *period being reported*.  Each
    10-K typically restates the two prior years, so a 2023 10-K will contain
    entries with fy=2023 but end dates of 2021-12-31, 2022-12-31, and
    2023-12-31.  Filtering on `fy` therefore returns wrong/missing data for
    all years except the current filing year.  The correct key is:

        fiscal_year = int(entry["end"][:4])   # year of the period end date

    When multiple filings cover the same period (e.g. an amendment or a later
    10-K restating a prior year), we keep the value from the most recently
    *filed* document so that any restatement is picked up automatically.

    Dimension handling (fixes companies like NOW / ServiceNow):
      1. First pass: collect only undimensioned entries (no "segment" key).
         These are the consolidated totals and are preferred.
      2. If no undimensioned total exists for a given (fiscal_year, fp), fall
         back to summing the dimensioned member entries that share the same
         (accn, end, fp) context.  This reconstructs the total for companies
         that never file an undimensioned aggregate (e.g. NOW splits revenue
         into LicenseAndServiceMember + TechnologyServiceMember only).

    For instant (balance-sheet) metrics fp is inferred from the form type:
      10-K  -> "FY"
      10-Q  -> quarterly fp taken directly from the entry
    For flow metrics fp is taken directly from the EDGAR entry.
    """
    VALID_FP = {"FY", "Q1", "Q2", "Q3", "Q4"}
    ANNUAL_FORMS  = {"10-K", "10-K405", "10-KT"}
    QUARTERLY_FORMS = {"10-Q", "10-QSB"}

    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tag_candidates:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            continue

        units = tag_data.get("units", {})
        if "EarningsPer" in tag:
            unit_key = "USD/shares"
        elif metric_name == "shares_outstanding":
            unit_key = "shares"
        else:
            unit_key = "USD"

        filings = units.get(unit_key, [])
        if not filings:
            for v in units.values():
                if v:
                    filings = v
                    break

        if not filings:
            continue

        # ------------------------------------------------------------------
        # Normalise each entry: derive true fiscal_year from end date, resolve
        # fp, filter to requested range and valid form types.
        # Split into undimensioned vs dimensioned buckets.
        # ------------------------------------------------------------------

        # undim_best  : (fiscal_year, fp) -> (val, filed_date)
        #   — later filed_date wins so restatements are picked up
        # dim_buckets : (fiscal_year, fp, accn, end) -> [val, ...]
        undim_best: dict[tuple[int, str], tuple[float, str]] = {}
        dim_buckets: dict[tuple[int, str, str, str], tuple[list[float], str]] = {}

        for entry in filings:
            form  = entry.get("form", "")
            fp    = entry.get("fp", "")
            val   = entry.get("val")
            end   = entry.get("end", "")    # "YYYY-MM-DD" — period end date
            accn  = entry.get("accn", "")
            filed = entry.get("filed", "")  # "YYYY-MM-DD" — filing date

            if val is None or not end:
                continue

            # ---- derive true fiscal year ----
            # Use the period end-date year for BOTH instant and flow metrics.
            # The `fy` EDGAR field represents the fiscal year of the *filing*,
            # not the period — and for non-Dec fiscal year companies (FICO ends
            # Sept 30, ADSK/CDNS end ~Jan, PAYX ends May 31, ADP ends June 30)
            # it can differ from the end-date year by up to 1 year.
            # Using end-date year universally gives the correct calendar period.
            try:
                fiscal_year = int(end[:4])
            except (ValueError, TypeError):
                continue

            if fiscal_year < start_fy or fiscal_year > end_fy:
                continue

            # ---- resolve fp for instant (balance-sheet) metrics ----
            if is_instant:
                if form in ANNUAL_FORMS:
                    fp = "FY"
                elif form in QUARTERLY_FORMS:
                    if fp not in {"Q1", "Q2", "Q3", "Q4"}:
                        continue
                else:
                    continue
            else:
                if fp not in VALID_FP:
                    continue
                # For flow metrics, only accept annual forms for FY entries
                # to avoid picking up a YTD figure from a 10-Q
                if fp == "FY" and form not in ANNUAL_FORMS:
                    continue

            key = (fiscal_year, fp)

            if _is_dimensioned(entry):
                bucket_key = (fiscal_year, fp, accn, end)
                if bucket_key not in dim_buckets:
                    dim_buckets[bucket_key] = ([], filed)
                dim_buckets[bucket_key][0].append(val)
            else:
                existing = undim_best.get(key)
                if is_instant:
                    # Prefer the EARLIEST filing that covers this period
                    # (contemporaneous 10-K). Later filings may restate values
                    # due to corporate actions like stock splits or restatements
                    # that we don't want to pick up for balance-sheet items.
                    if existing is None or filed < existing[1]:
                        undim_best[key] = (val, filed, True)
                else:
                    # For flow metrics, prefer most recently filed (latest
                    # restatement wins — captures corrections and amendments).
                    if existing is None or filed > existing[1]:
                        undim_best[key] = (val, filed, False)

        # ------------------------------------------------------------------
        # Build result: prefer undimensioned totals; sum dimensions as fallback
        # ------------------------------------------------------------------
        result: dict[tuple[int, str], float] = {
            k: v for k, (v, _, _c) in undim_best.items()
        }

        # For any (fiscal_year, fp) with no undimensioned total, sum the
        # dimension members from the most recently filed context.
        if dim_buckets:
            by_period: dict[tuple[int, str], list[tuple[tuple, list[float], str]]] = defaultdict(list)
            for (fy, fp, accn, end), (vals, filed) in dim_buckets.items():
                by_period[(fy, fp)].append(((fy, fp, accn, end), vals, filed))

            for period_key, buckets in by_period.items():
                if period_key in result:
                    continue  # already have undimensioned total
                # Pick the bucket from the most recently filed document
                best_bucket = max(buckets, key=lambda x: (x[2], len(x[1])))
                result[period_key] = sum(best_bucket[1])

        if result:
            # Only accept this tag if it produced at least one annual (FY) entry.
            # Some tags have quarterly data in the requested range but no annual
            # entries (e.g. VRSN's NetIncomeLoss has Q1-Q3 10-Q entries post-2012
            # but no 10-K FY entries — they switched to ProfitLoss for annuals).
            # Returning quarterly-only data here would block the fallback to the
            # correct annual tag, so we require at least one FY key.
            if any(fp == "FY" for _, fp in result):
                return result
            # Has only quarterly data — continue to next tag candidate

    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fundamentals(
    cik: str | int,
    ticker: str,
    start_fy: int,
    end_fy: int,
    metrics: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Retrieve fundamental data for a company from SEC EDGAR.

    Returns a flat DataFrame with one row per (ticker, cik, fiscal_year, period)
    combination, making it easy to merge with price or macro data on those keys.

    Parameters
    ----------
    cik : str or int
        SEC Central Index Key.  Padded or unpadded, int or string.
    ticker : str
        Exchange ticker symbol (e.g. "AAPL").  Written into every row of the
        returned DataFrame so it can be used as a merge key.
    start_fy : int
        First fiscal year to include (e.g. 2015).
    end_fy : int
        Last fiscal year to include (e.g. 2023).
    metrics : dict, optional
        Custom metric dict (same structure as FUNDAMENTAL_METRICS).
        Defaults to FUNDAMENTAL_METRICS.

    Returns
    -------
    pd.DataFrame
        Columns:
          ticker        – the ticker symbol passed in
          cik           – zero-padded 10-digit CIK (str)
          fiscal_year   – integer fiscal year (e.g. 2022)
          period        – "FY", "Q1", "Q2", or "Q3"
          <statement>_<metric>  – one flat column per metric

        The (ticker, cik, fiscal_year, period) tuple is a natural composite
        merge key.

    Examples
    --------
    >>> df = get_fundamentals("0000320193", "AAPL", 2018, 2023)
    >>> df.query("period == 'FY'")[["fiscal_year", "income_statement_total_revenue"]]
    """
    if metrics is None:
        metrics = config.FUNDAMENTAL_METRICS  # use module-level definition; do NOT reference config here

    cik_padded = _normalize_cik(cik)
    facts = _fetch_company_facts(cik_padded)
    _facts = facts  # keep reference for computed fallbacks below

    # Collect all (fy, fp) -> metric_value mappings
    # Structure: { (fy, fp): { "statement_metric": value } }
    period_data: dict[tuple[int, str], dict[str, float | None]] = {}

    for statement, metric_dict in metrics.items():
        for metric_name, tag_candidates in metric_dict.items():
            is_instant = metric_name in INSTANTANEOUS_METRICS
            col_name = f"{statement}_{metric_name}"

            period_map = _extract_period_values(
                facts,
                tag_candidates,
                metric_name,
                start_fy,
                end_fy,
                is_instant,
            )

            for (fy, fp), val in period_map.items():
                key = (fy, fp)
                if key not in period_data:
                    period_data[key] = {}
                period_data[key][col_name] = val

    if not period_data:
        all_cols = [metric for _, metric_dict in metrics.items() for metric in metric_dict]
        return pd.DataFrame(columns=["ticker", "cik", "fiscal_year", "period"] + all_cols)

    # Build flat rows (use prefixed col names internally; rename happens below)
    all_metric_cols = [
        f"{stmt}_{m}"
        for stmt, md in metrics.items()
        for m in md
    ]

    rows = []
    for (fy, fp), metric_vals in period_data.items():
        row: dict = {
            "ticker": ticker,
            "cik": cik_padded,
            "fiscal_year": fy,
            "period": fp,
        }
        for col in all_metric_cols:
            row[col] = metric_vals.get(col)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Rename columns: drop the "statement_" prefix so each column is exactly
    # the metric key as defined in the config (e.g. "Revenues", "Assets").
    rename_map = {
        f"{stmt}_{metric}": metric
        for stmt, metric_dict in metrics.items()
        for metric in metric_dict
    }
    df = df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # Computed fallbacks
    # ------------------------------------------------------------------

    # GrossProfit: some companies (CDNS, WDAY, VRSN, INTU post-2020,
    # ADP post-2020, FICO post-2021) never tag GrossProfit directly.
    # Derive it as Revenues - CostOfRevenues where both are available.
    if "GrossProfit" in df.columns and "Revenues" in df.columns and "CostOfRevenues" in df.columns:
        mask = df["GrossProfit"].isna() & df["Revenues"].notna() & df["CostOfRevenues"].notna()
        df.loc[mask, "GrossProfit"] = df.loc[mask, "Revenues"] - df.loc[mask, "CostOfRevenues"]

    # Liabilities: CDNS and ADSK never tag the total Liabilities line —
    # they only file LiabilitiesCurrent and LiabilitiesNoncurrent separately.
    # Sum them when the total is missing.
    if "Liabilities" in df.columns:
        liab_cur_col  = "income_statement_LiabilitiesCurrent"   # won't exist
        # We need to re-extract LiabilitiesCurrent + LiabilitiesNoncurrent
        # directly from facts and add them here.
        if df["Liabilities"].isna().any():
            for tag_pair in [("LiabilitiesCurrent", "LiabilitiesNoncurrent")]:
                cur_map  = _extract_period_values(_facts, [tag_pair[0]], tag_pair[0], start_fy, end_fy, True)
                non_map  = _extract_period_values(_facts, [tag_pair[1]], tag_pair[1], start_fy, end_fy, True)
                if cur_map and non_map:
                    for idx, row_data in df.iterrows():
                        if pd.isna(row_data["Liabilities"]):
                            key = (int(row_data["fiscal_year"]), row_data["period"])
                            cur_val = cur_map.get(key)
                            non_val = non_map.get(key)
                            if cur_val is not None and non_val is not None:
                                df.at[idx, "Liabilities"] = cur_val + non_val

    # Sort and set a logical index
    period_order = {"FY": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    df["_period_sort"] = df["period"].map(lambda p: period_order.get(p, 99))
    df = (
        df.sort_values(["fiscal_year", "_period_sort"])
          .drop(columns=["_period_sort"])
          .reset_index(drop=True)
    )

    return df