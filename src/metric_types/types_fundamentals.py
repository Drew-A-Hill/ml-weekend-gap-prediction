"""
File: types_fundamentals.py
Author: Drew Hill
This file is used to hold argument typing schema.
"""
from typing import TypedDict

class FundamentalFlags(TypedDict):
    revenue: bool
    cost_of_revenue: bool
    gross_profit: bool
    operating_expenses: bool
    research_and_development: bool
    selling_general_admin: bool
    operating_income: bool
    net_income: bool
    income_tax_expense: bool
    interest_expense: bool
    eps_basic: bool
    eps_diluted: bool
    shares_basic: bool
    shares_diluted: bool

    assets: bool
    current_assets: bool
    cash: bool
    accounts_receivable: bool
    inventory: bool
    property_plant_equipment: bool
    goodwill: bool
    intangible_assets: bool
    liabilities: bool
    current_liabilities: bool
    accounts_payable: bool
    short_term_debt: bool
    long_term_debt: bool
    equity: bool
    retained_earnings: bool

    operating_cash_flow: bool
    investing_cash_flow: bool
    financing_cash_flow: bool
    capex: bool
    depreciation: bool
    stock_based_compensation: bool
    dividends_paid: bool