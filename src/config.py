"""

"""
from pathlib import Path

HEADER: dict[str, str] = {
    "User-Agent": "DrewHill hill.dr@northeastern.edu"
}

DATA_DIR = Path(__file__).resolve().parents[1] / "structured_csv_data_files"

# -------------------------------------------------- Filtering --------------------------------------------------
PERIOD: str = "13y"
INTERVAL: str = "1d"

MIN_CAP: int = 10_000_000_000
MAX_CAP: int = 200_000_000_000
MIN_PROFIT_MARGIN: float = 0.0001
MIN_PUBLIC_AGE: int = 12

EXCHANGE: list[str] = [
    "NMS",
    "NYQ",
    "ASE"
]

SECTORS: list[str] = [
    "Technology"
]

INDUSTRIES: list[str] = [
    "Software - Application",
    "Software - Infrastructure"
]

FINANCIAL_METRICS: dict[str, str] = {
    "open_p": "Open",
    "close_p": "Close",
    "high_p": "High",
    "low_p": "Low",
    "volume": "Volume",
    "dividend": "Dividends",
    "stock_splits": "Stock Splits"
}


# Fundamental Metrics
ANNUAL_FORMS    = {"10-K", "10-K405", "10-KT"}
QUARTERLY_FORMS = {"10-Q", "10-QSB"}
VALID_FP        = {"FY", "Q1", "Q2", "Q3", "Q4"}

SINGLE_Q_MIN = 78
SINGLE_Q_MAX = 105

FUNDAMENTAL_METRICS = {
    "revenues": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "cost_of_revenues": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
        "CostOfSales",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "assets": [
        "Assets",
    ],
    "liabilities": [
        "Liabilities",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "shares": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ],
}

BS_METRICS = {"assets", "liabilities", "equity", "shares"}
IS_METRICS    = {"revenues", "cost_of_revenues", "net_income"}