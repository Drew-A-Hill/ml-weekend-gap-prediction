"""

"""
HEADER: dict[str, str] = {
    "User-Agent": "DrewHill hill.dr@northeastern.edu"
}

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

FUNDAMENTAL_METRICS = {
    "income_statement": {
        "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "cost_of_revenue": "CostOfRevenue",
        "gross_profit": "GrossProfit",
        "operating_expenses": "OperatingExpenses",
        "research_and_development": "ResearchAndDevelopmentExpense",
        "selling_general_admin": "SellingGeneralAndAdministrativeExpense",
        "operating_income": "OperatingIncomeLoss",
        "net_income": "NetIncomeLoss",
        "income_tax_expense": "IncomeTaxExpenseBenefit",
        "eps_basic": "EarningsPerShareBasic",
        "eps_diluted": "EarningsPerShareDiluted"
    },

    "balance_sheet": {
        "assets": "Assets",
        "current_assets": "AssetsCurrent",
        "cash": "CashAndCashEquivalentsAtCarryingValue",
        "accounts_receivable": "AccountsReceivableNetCurrent",
        "property_plant_equipment": "PropertyPlantAndEquipmentNet",
        "liabilities": "Liabilities",
        "current_liabilities": "LiabilitiesCurrent",
        "accounts_payable": "AccountsPayableCurrent",
        "long_term_debt": "LongTermDebt",
        "equity": "StockholdersEquity",
        "retained_earnings": "RetainedEarningsAccumulatedDeficit"
    },

    "cash_flow": {
        "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        "investing_cash_flow": "NetCashProvidedByUsedInInvestingActivities",
        "financing_cash_flow": "NetCashProvidedByUsedInFinancingActivities",
        "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
        "depreciation": "DepreciationDepletionAndAmortization",
        "stock_based_compensation": "ShareBasedCompensation"
    }
}