"""Per-fold preprocessing for tabular features (LogReg / XGBoost / PCA).

Reproduces the exact pipeline from the original notebooks:
- OBV: per-ticker z-score (removes cumulative scale, keeps direction)
- LOG1P_FEATURES: log1p then StandardScaler
- Binary missingness indicators: untouched
- All other numeric features: StandardScaler only

Scaler is fitted on the training fold only.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

LOG1P_FEATURES = ['ATR', 'BollingerBandWidth', 'FiveDStdDev', 'VolumeRatio']
BINARY_FEATURES = ['GrossMargin_missing', 'CostOfRevenues_missing', 'Liabilities_missing']


def preprocess_tabular(X_train, X_test, train_tickers, test_tickers, features):
    """Fit on X_train only, transform both. Returns (X_tr, X_te) as np.float32."""
    X_tr, X_te = X_train.copy(), X_test.copy()

    obv_stats = (X_tr.assign(Ticker=train_tickers.values)
                     .groupby('Ticker')['OBV']
                     .agg(['mean', 'std']))
    for df_x, tickers in [(X_tr, train_tickers), (X_te, test_tickers)]:
        means = tickers.map(obv_stats['mean']).fillna(obv_stats['mean'].mean())
        stds = tickers.map(obv_stats['std']).fillna(obv_stats['std'].mean()).replace(0, 1)
        df_x['OBV'] = (df_x['OBV'].values - means.values) / stds.values

    for col in LOG1P_FEATURES:
        X_tr[col] = np.log1p(X_tr[col].clip(lower=0))
        X_te[col] = np.log1p(X_te[col].clip(lower=0))

    cols_to_scale = [f for f in features if f not in BINARY_FEATURES]
    scaler = StandardScaler()
    X_tr[cols_to_scale] = scaler.fit_transform(X_tr[cols_to_scale])
    X_te[cols_to_scale] = scaler.transform(X_te[cols_to_scale])
    return X_tr.values, X_te.values


def global_preprocess_tabular(X, tickers, features):
    """Same transforms as preprocess_tabular but fitted on the full dataset.

    Used only for the global PCA experiment (leakage upper bound).
    """
    X_proc = X.copy()
    obv_stats = (X_proc.assign(Ticker=tickers.values)
                       .groupby('Ticker')['OBV']
                       .agg(['mean', 'std']))
    means = tickers.map(obv_stats['mean']).fillna(obv_stats['mean'].mean())
    stds = tickers.map(obv_stats['std']).fillna(obv_stats['std'].mean()).replace(0, 1)
    X_proc['OBV'] = (X_proc['OBV'].values - means.values) / stds.values

    for col in LOG1P_FEATURES:
        X_proc[col] = np.log1p(X_proc[col].clip(lower=0))

    cols_to_scale = [f for f in features if f not in BINARY_FEATURES]
    scaler = StandardScaler()
    X_proc[cols_to_scale] = scaler.fit_transform(X_proc[cols_to_scale])
    return X_proc.values
