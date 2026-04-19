"""LSTM architecture, sequence builder, per-fold training and walk-forward CV.

Reproduces the exact logic from `lstm_auto_correlated_metric.ipynb`:
- per-ticker sliding-window sequence construction (year of target row =
  fold membership)
- single-layer LSTM + last-timestep head + sigmoid
- rolling 3-year walk-forward with hyperparameter grid search on inner val
- permutation feature importance (AUC drop averaged over 5 shuffles)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_PARAM_GRID = [
    {'hidden_size': h, 'dropout': d}
    for h in [32, 64] for d in [0.2, 0.3]
]


# ── Architecture ─────────────────────────────────────────────────────────────

class LSTMGapPredictor(nn.Module):
    """Single-layer LSTM, last-timestep hidden state → linear → sigmoid."""

    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


# ── Training / evaluation primitives ────────────────────────────────────────

def make_loader(X, y, batch_size=64, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        proba = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    preds = (proba >= 0.5).astype(int)
    return roc_auc_score(y, proba), accuracy_score(y, preds), proba


def fit_model(X_train, y_train, X_val, y_val,
              hidden_size, dropout, lr=0.001,
              max_epochs=50, patience=10, batch_size=64):
    """Fit LSTM with early stopping on val AUC. Returns (model, best_val_auc)."""
    model = LSTMGapPredictor(X_train.shape[2], hidden_size, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    loader = make_loader(X_train, y_train, batch_size=batch_size)
    best_auc, best_state, wait = -np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        val_auc, _, _ = evaluate(model, X_val, y_val)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc


# ── Sequence construction ───────────────────────────────────────────────────

def build_sequences(df, features, target, window=4, return_dates=False):
    """Per-ticker sliding-window sequences.

    Each window uses rows [i-window+1 .. i] as input, row i as target.
    Year of row i → fold membership. Returns (X, y, years, extreme_flags)
    or (X, y, years, extreme_flags, target_dates) if return_dates=True.
    """
    X_list, y_list, yr_list, ex_list, dt_list = [], [], [], [], []
    has_extreme = 'is_extreme_event' in df.columns

    for _, grp in df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        feat_vals = grp[features].values.astype(np.float32)
        tgt_vals = grp[target].values.astype(np.float32)
        yr_vals = grp['Year'].values
        dt_vals = grp['Date'].values
        ex_vals = (grp['is_extreme_event'].values if has_extreme
                   else np.zeros(len(grp), dtype=int))

        for i in range(window - 1, len(grp)):
            X_list.append(feat_vals[i - window + 1: i + 1])
            y_list.append(tgt_vals[i])
            yr_list.append(yr_vals[i])
            ex_list.append(ex_vals[i])
            dt_list.append(dt_vals[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    yr = np.array(yr_list)
    ex = np.array(ex_list)
    if return_dates:
        return X, y, yr, ex, np.array(dt_list)
    return X, y, yr, ex


def chronological_sort(X, y, years, dates, *others):
    """Sort sequence arrays by target date. Required before tscv."""
    order = np.argsort(dates)
    sorted_others = tuple(arr[order] for arr in others)
    return (X[order], y[order], years[order], dates[order]) + sorted_others


# ── Walk-forward CV (year folds, rolling 3-year LSTM variant) ───────────────

def run_walkforward(X_all, y_all, years_all, folds_def,
                    param_grid=None, label='', transform=None, verbose=True):
    """Per-fold tune-then-refit walk-forward. Mirrors LSTM notebook `run_walkforward`.

    Returns dict with results_df (per-fold metrics), models, Xts, yts, probas, params.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    sc = RobustScaler()
    fold_results = []
    fold_models, fold_Xts, fold_yts, fold_probas, fold_params = [], [], [], [], []
    fold_cms, fold_rocs = [], []

    for fold in folds_def:
        test_yr = fold['test_year']
        train_yrs = fold['train_years']
        iv_yr = max(train_yrs)
        it_yrs = [y for y in train_yrs if y != iv_yr]

        mi = np.isin(years_all, it_yrs)
        vi = years_all == iv_yr
        mf = np.isin(years_all, train_yrs)
        mt = years_all == test_yr

        nf = X_all.shape[2]
        sc.fit(X_all[mf].reshape(-1, nf))

        def prep(X):
            s = sc.transform(X.reshape(-1, nf)).reshape(X.shape).astype(np.float32)
            return transform(s) if transform else s

        Xi, yi = prep(X_all[mi]), y_all[mi]
        Xv, yv = prep(X_all[vi]), y_all[vi]
        Xf, yf = prep(X_all[mf]), y_all[mf]
        Xt, yt = prep(X_all[mt]), y_all[mt]

        best_auc, best_p = -np.inf, None
        for p in param_grid:
            torch.manual_seed(42)
            _, va = fit_model(Xi, yi, Xv, yv, **p)
            if va > best_auc:
                best_auc, best_p = va, p.copy()

        torch.manual_seed(42)
        model, _ = fit_model(Xf, yf, Xv, yv, **best_p)
        auc, acc, proba = evaluate(model, Xt, yt)
        preds = (proba >= 0.5).astype(int)

        cm = confusion_matrix(yt, preds)
        fpr, tpr, _ = roc_curve(yt, proba)

        fold_results.append({
            'Test year': test_yr,
            'AUC-ROC': round(auc, 4),
            'Accuracy': round(acc, 4),
            'Precision': round(precision_score(yt, preds, zero_division=0), 4),
            'Recall': round(recall_score(yt, preds, zero_division=0), 4),
            'F1': round(f1_score(yt, preds, zero_division=0), 4),
            'Baseline': round(max(yt.mean(), 1 - yt.mean()), 4),
        })
        fold_models.append(model)
        fold_Xts.append(Xt)
        fold_yts.append(yt)
        fold_probas.append(proba)
        fold_params.append(best_p)
        fold_cms.append(cm)
        fold_rocs.append((fpr, tpr))

        if verbose and label:
            print(f'  [{label}] {test_yr} | h={best_p["hidden_size"]} '
                  f'dr={best_p["dropout"]} | AUC={auc:.3f}')

    results_df = pd.DataFrame(fold_results).set_index('Test year')
    return dict(results_df=results_df,
                fold_results=fold_results,
                fold_models=fold_models,
                fold_Xts=fold_Xts, fold_yts=fold_yts,
                fold_probas=fold_probas,
                fold_params=fold_params,
                fold_cms=fold_cms, fold_rocs=fold_rocs)


# ── Walk-forward CV (TimeSeriesSplit — index-based, for cross-validated best) ──

def run_walkforward_tscv(X_all, y_all, tscv_folds, param_grid=None,
                          transform=None, label='', verbose=True):
    """Same training loop but over TimeSeriesSplit folds (no calendar years).

    Inner validation: last 20% of training indices per fold.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    sc = RobustScaler()
    aucs, accs, precs, recs, f1s = [], [], [], [], []
    for fold_i, (tr_idx, te_idx) in enumerate(tscv_folds, 1):
        split_pt = int(len(tr_idx) * 0.8)
        itr_idx, iv_idx = tr_idx[:split_pt], tr_idx[split_pt:]

        nf = X_all.shape[2]
        sc.fit(X_all[tr_idx].reshape(-1, nf))

        def prep(X):
            s = sc.transform(X.reshape(-1, nf)).reshape(X.shape).astype(np.float32)
            return transform(s) if transform else s

        Xi, yi = prep(X_all[itr_idx]), y_all[itr_idx]
        Xv, yv = prep(X_all[iv_idx]), y_all[iv_idx]
        Xf, yf = prep(X_all[tr_idx]), y_all[tr_idx]
        Xt, yt = prep(X_all[te_idx]), y_all[te_idx]

        best_auc, best_p = -np.inf, None
        for p in param_grid:
            torch.manual_seed(42)
            _, va = fit_model(Xi, yi, Xv, yv, **p)
            if va > best_auc:
                best_auc, best_p = va, p.copy()

        torch.manual_seed(42)
        model, _ = fit_model(Xf, yf, Xv, yv, **best_p)
        auc, acc, proba = evaluate(model, Xt, yt)
        preds = (proba >= 0.5).astype(int)
        aucs.append(auc)
        accs.append(acc)
        precs.append(precision_score(yt, preds, zero_division=0))
        recs.append(recall_score(yt, preds, zero_division=0))
        f1s.append(f1_score(yt, preds, zero_division=0))

        if verbose and label:
            print(f'  [{label} tscv] fold {fold_i} | AUC={auc:.3f}')

    return dict(aucs=aucs, accs=accs, precisions=precs, recalls=recs, f1s=f1s)


# ── Permutation feature importance (per-fold, 5 shuffles, AUC drop) ─────────

def permutation_importance(fold_models, fold_Xts, fold_yts, fold_results,
                           features, n_repeats=5, seed=42):
    """Shuffle each feature across sequences; measure drop in AUC.

    Returns DataFrame indexed by feature, columns = test_years + 'Avg AUC drop'.
    """
    rng = np.random.default_rng(seed)
    fold_importances = []
    for model, Xt, yt, row in zip(fold_models, fold_Xts, fold_yts, fold_results):
        base_auc = row['AUC-ROC']
        perm_imp = {}
        for fi, feat in enumerate(features):
            drops = []
            for _ in range(n_repeats):
                Xp = Xt.copy()
                Xp[:, :, fi] = Xp[rng.permutation(Xp.shape[0]), :, fi]
                drops.append(evaluate(model, Xp, yt)[0])
            perm_imp[feat] = round(base_auc - float(np.mean(drops)), 5)
        fold_importances.append(perm_imp)

    imp_df = pd.DataFrame(fold_importances).T
    imp_df.columns = [r['Test year'] for r in fold_results]
    imp_df['Avg AUC drop'] = imp_df.mean(axis=1)
    return imp_df.sort_values('Avg AUC drop', ascending=False)
