"""Fold definitions used across notebooks.

Two orthogonal validation strategies:
- `build_year_folds_tabular`  — expanding calendar-year folds (LogReg / XGBoost / PCA).
- `build_year_folds_lstm`     — rolling 3-year folds used by LSTM notebook.
- `build_tscv_folds`          — TimeSeriesSplit(n_splits) over a date-sorted dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# ── Tabular year folds (expanding window, identical to logistic_regression.ipynb) ──
TABULAR_FOLDS_DEF = [
    {'train_years': list(range(2016, 2019)), 'test_year': 2019},
    {'train_years': list(range(2016, 2020)), 'test_year': 2020},
    {'train_years': list(range(2016, 2021)), 'test_year': 2021},
    {'train_years': list(range(2016, 2022)), 'test_year': 2022},
    {'train_years': list(range(2016, 2023)), 'test_year': 2023},
    {'train_years': list(range(2016, 2024)), 'test_year': 2024},
]

# ── LSTM rolling 3-year folds ──
LSTM_FOLDS_DEF = [
    {'train_years': [2016, 2017, 2018], 'test_year': 2019},
    {'train_years': [2017, 2018, 2019], 'test_year': 2020},
    {'train_years': [2018, 2019, 2020], 'test_year': 2021},
    {'train_years': [2019, 2020, 2021], 'test_year': 2022},
    {'train_years': [2020, 2021, 2022], 'test_year': 2023},
    {'train_years': [2021, 2022, 2023], 'test_year': 2024},
]

COVID_YEAR = 2020


def build_year_folds_tabular(primary_df):
    """Return list of (train_idx, test_idx) tuples from TABULAR_FOLDS_DEF."""
    folds = []
    for fd in TABULAR_FOLDS_DEF:
        tr = np.where(primary_df['Year'].isin(fd['train_years']))[0]
        te = np.where(primary_df['Year'] == fd['test_year'])[0]
        folds.append((tr, te))
    return folds, TABULAR_FOLDS_DEF


def build_tscv_folds(X, n_splits=18):
    """Return (folds, tscv_obj) where folds is a list of (train_idx, test_idx)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X)), tscv


def non_covid_indices(test_years):
    """Indices of folds whose test year is not the COVID year."""
    return [i for i, yr in enumerate(test_years) if yr != COVID_YEAR]


def non_covid_mean(values, test_years):
    """Average `values` excluding the COVID fold.

    If `test_years` is falsy (tscv has no calendar labels) return the full mean.
    """
    if not test_years:
        return float(np.mean(values))
    nc = non_covid_indices(test_years)
    return float(np.mean([values[i] for i in nc]))
