"""

"""
from pathlib import Path

HEADER: dict[str, str] = {
    "User-Agent": "DrewHill hill.dr@northeastern.edu"
}
DATA_DIR = Path(__file__).resolve().parents[1] / "structured_csv_data_files"

PERIOD: str = "10y"
INTERVAL: str = "1d"

MIN_CAP: int = 10_000_000_000
MAX_CAP: int = 200_000_000_000
MIN_PROFIT_MARGIN: float = 0.01
MIN_PUBLIC_AGE: int = 15

EXCHANGE: list[str] = [
    "NMS",
    "NYQ",
    "ASE"
]

INDUSTRIES: list[str] = [
    "Semiconductors",
    "Software",
    "Consumer Electronics",
    "Semiconductor Equipment"
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

YEARS: list[str] = [
    "2026",
    "2025",
    "2024",
    "2023",
    "2022",
    "2021",
    "2019",
    "2018",
    "2017",
    "2016"
]

QUARTERS: list[str] = [
    "Q1",
    "Q2",
    "Q3",
    "Q4"
]

FUNDAMENTAL_METRICS = {

    "income_statement": {
        "revenue": [
            # "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues"
        ],
        "cost_of_revenue": [
            "CostOfRevenue"
        ],
        "gross_profit": [
            "GrossProfit"
        ],
        "operating_expenses": [
            "OperatingExpenses"
        ],
        "research_and_development": [
            "ResearchAndDevelopmentExpense"
        ],
        "selling_general_admin": [
            "SellingGeneralAndAdministrativeExpense"
        ],
        "operating_income": [
            "OperatingIncomeLoss"
        ],
        "net_income": [
            "NetIncomeLoss",
            "ProfitLoss"
        ],
        "income_tax_expense": [
            "IncomeTaxExpenseBenefit"
        ],
        "interest_expense": [
            "InterestExpense"
        ],
        "eps_basic": [
            "EarningsPerShareBasic"
        ],
        "eps_diluted": [
            "EarningsPerShareDiluted"
        ],
        "shares_basic": [
            "WeightedAverageNumberOfSharesOutstandingBasic"
        ],
        "shares_diluted": [
            "WeightedAverageNumberOfDilutedSharesOutstanding"
        ]
    },

    "balance_sheet": {
        "assets": [
            "Assets"
        ],
        "current_assets": [
            "AssetsCurrent"
        ],
        "cash": [
            "CashAndCashEquivalentsAtCarryingValue"
        ],
        "accounts_receivable": [
            "AccountsReceivableNetCurrent"
        ],
        "inventory": [
            "InventoryNet"
        ],
        "property_plant_equipment": [
            "PropertyPlantAndEquipmentNet"
        ],
        "goodwill": [
            "Goodwill"
        ],
        "intangible_assets": [
            "IntangibleAssetsNetExcludingGoodwill"
        ],
        "liabilities": [
            "Liabilities"
        ],
        "current_liabilities": [
            "LiabilitiesCurrent"
        ],
        "accounts_payable": [
            "AccountsPayableCurrent"
        ],
        "short_term_debt": [
            "DebtCurrent"
        ],
        "long_term_debt": [
            "LongTermDebt",
            "LongTermDebtNoncurrent"
        ],
        "equity": [
            "StockholdersEquity"
        ],
        "retained_earnings": [
            "RetainedEarningsAccumulatedDeficit"
        ]
    },

    "cash_flow": {
        "operating_cash_flow": [
            "NetCashProvidedByUsedInOperatingActivities"
        ],
        "investing_cash_flow": [
            "NetCashProvidedByUsedInInvestingActivities"
        ],
        "financing_cash_flow": [
            "NetCashProvidedByUsedInFinancingActivities"
        ],
        "capex": [
            "PaymentsToAcquirePropertyPlantAndEquipment"
        ],
        "depreciation": [
            "DepreciationDepletionAndAmortization"
        ],
        "stock_based_compensation": [
            "ShareBasedCompensation"
        ],
        "dividends_paid": [
            "PaymentsOfDividends"
        ]
    }
}