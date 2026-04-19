"""Dataset loading + feature groups.

Two feature sets are defined:
- `TABULAR_FEATURES` (18): the LogReg / XGBoost / PCA feature set — technicals,
  fundamentals, and missingness indicators.
- `LSTM_FEATURES` (14): technicals only. Fundamentals are excluded because
  quarterly values repeat across ~13 consecutive weekly rows, giving the LSTM
  gate mechanism no temporal signal.
"""
from __future__ import annotations

import pandas as pd

DATASET_PATH = '../structured_csv_data_files/fetched_data/dataset_clean.csv'

# ── Tabular feature groups (used by LogReg / XGBoost / PCA notebook) ──
MOMENTUM_TAB    = ['MACD', 'ROC', 'StochPercK']
TREND_TAB       = ['CloseVEma50', 'CloseVSma20', 'ADX']
VOLATILITY_TAB  = ['BollingerBandWidth', 'ATR', 'FiveDStdDev']
VOLUME_TAB      = ['OBV', 'MFI', 'VolumeRatio']
FUNDAMENTAL_TAB = ['NetMargin', 'RoA', 'RevGrowthQoQ']
MISS_IND_TAB    = ['GrossMargin_missing', 'CostOfRevenues_missing', 'Liabilities_missing']

TABULAR_FEATURES = (MOMENTUM_TAB + TREND_TAB + VOLATILITY_TAB +
                    VOLUME_TAB + FUNDAMENTAL_TAB + MISS_IND_TAB)

# ── XGBoost feature set (21) — canonical xgboost_model.ipynb ──
# Adds RSI, GrossMargin, WeeklyRange back (collinearity doesn't hurt trees).
XGBOOST_FEATURES = (
    ['RSI'] + MOMENTUM_TAB +                          # momentum (4)
    TREND_TAB +                                       # trend (3)
    VOLATILITY_TAB +                                  # volatility (3)
    VOLUME_TAB +                                      # volume (3)
    ['GrossMargin'] + FUNDAMENTAL_TAB +               # fundamentals (4)
    ['WeeklyRange'] +                                 # price agg (1)
    MISS_IND_TAB                                      # missingness (3)
)
assert len(XGBOOST_FEATURES) == 21, f'Expected 21, got {len(XGBOOST_FEATURES)}'

# ── LSTM feature groups (14 technical features — no fundamentals) ──
MOMENTUM_LSTM   = ['RSI', 'MACD', 'ROC', 'StochPercK']
TREND_LSTM      = ['CloseVEma50', 'CloseVSma20', 'ADX']
VOLATILITY_LSTM = ['BollingerBandWidth', 'ATR', 'FiveDStdDev']
VOLUME_LSTM     = ['OBV', 'MFI', 'VolumeRatio']
PRICE_AGG_LSTM  = ['WeeklyRange']

LSTM_FEATURES = (MOMENTUM_LSTM + TREND_LSTM + VOLATILITY_LSTM +
                 VOLUME_LSTM + PRICE_AGG_LSTM)

TARGET = 'GapUp'


def load_dataset(path: str = DATASET_PATH):
    """Load raw dataset and split into primary / extreme event frames.

    Primary frame is date-sorted (required for TimeSeriesSplit chronological
    slicing). Index is reset on both.
    """
    df_all = pd.read_csv(path, parse_dates=['Date'])
    primary = (df_all[df_all['is_extreme_event'] == 0]
               .sort_values('Date')
               .reset_index(drop=True))
    extreme = (df_all[df_all['is_extreme_event'] == 1]
               .reset_index(drop=True))
    return df_all, primary, extreme
