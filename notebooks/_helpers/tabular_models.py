"""Tabular modeling helpers — LogReg (baseline & PCA) and XGBoost (± PCA).

Every function fits its scaler / PCA / model on training folds only and tunes
hyperparameters on inner validation (last training year for year folds, last
20% of training indices for tscv folds). Returns per-fold AUC / Accuracy lists
plus artifacts (confusion matrices, ROC curves, feature importances) that the
notebooks store in pickles.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .preprocessing import preprocess_tabular, global_preprocess_tabular

C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
LR_DEFAULTS = dict(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)

# ── 48-combo XGBoost grid (matches canonical xgboost_model.ipynb) ──
XGB_PARAM_GRID = {
    'max_depth':        [3, 4, 5],
    'learning_rate':    [0.05, 0.1],
    'subsample':        [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'min_child_weight': [1, 3],
}
XGB_ES_DEFAULTS = dict(n_estimators=500, early_stopping_rounds=30,
                       eval_metric='auc', random_state=42, verbosity=0)

# Simple XGBoost recipe (used only for the n_splits sweep — fast, no grid)
XGB_SIMPLE = dict(n_estimators=500, learning_rate=0.05,
                  early_stopping_rounds=30, eval_metric='auc',
                  random_state=42, verbosity=0)


def _evaluate_fold(model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    cm = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    return dict(auc=auc, acc=acc, cm=cm, roc=(fpr, tpr),
                y_pred=y_pred, y_prob=y_prob)


# ── Inner-val helpers ────────────────────────────────────────────────────────

def _inner_split_year(primary, train_years):
    """Inner val = last training year. Returns (inner_train_idx, inner_val_idx)."""
    val_year = train_years[-1]
    inner_train = [y for y in train_years if y != val_year]
    itr_idx = np.where(primary['Year'].isin(inner_train))[0]
    iv_idx = np.where(primary['Year'] == val_year)[0]
    return itr_idx, iv_idx


def _inner_split_tscv(tr_idx, ratio=0.8):
    """Inner val = last (1-ratio) of training indices. Assumes chronologically sorted."""
    split_pt = int(len(tr_idx) * ratio)
    return tr_idx[:split_pt], tr_idx[split_pt:]


def _tune_C(X_itr, y_itr, X_iv, y_iv, C_grid=C_GRID):
    """Pick C with highest inner-val AUC."""
    best_C, best_auc = C_grid[0], -np.inf
    for c in C_grid:
        m = LogisticRegression(C=c, **LR_DEFAULTS)
        m.fit(X_itr, y_itr)
        a = roc_auc_score(y_iv, m.predict_proba(X_iv)[:, 1])
        if a > best_auc:
            best_auc, best_C = a, c
    return best_C, best_auc


# ── Logistic Regression: baseline (no PCA) ──────────────────────────────────

def run_logreg_year(primary, X_all, y_all, tickers_all, features, folds_def):
    """LogReg L1 with C tuned per fold via inner-val AUC (last training year)."""
    results = {'test_years': [], 'aucs': [], 'accs': [], 'cms': [], 'rocs': [],
               'best_cs': [], 'coef': []}
    for fold in folds_def:
        train_years = fold['train_years']
        test_year = fold['test_year']

        tr_idx = np.where(primary['Year'].isin(train_years))[0]
        te_idx = np.where(primary['Year'] == test_year)[0]
        itr_idx, iv_idx = _inner_split_year(primary, train_years)

        X_tr_s, X_te_s = preprocess_tabular(
            X_all.iloc[tr_idx], X_all.iloc[te_idx],
            tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
        X_itr_s, X_iv_s = preprocess_tabular(
            X_all.iloc[itr_idx], X_all.iloc[iv_idx],
            tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)

        best_C, _ = _tune_C(X_itr_s, y_all.iloc[itr_idx],
                             X_iv_s, y_all.iloc[iv_idx])

        model = LogisticRegression(C=best_C, **LR_DEFAULTS)
        model.fit(X_tr_s, y_all.iloc[tr_idx])
        fold_m = _evaluate_fold(model, X_te_s, y_all.iloc[te_idx])
        results['test_years'].append(test_year)
        results['aucs'].append(fold_m['auc'])
        results['accs'].append(fold_m['acc'])
        results['cms'].append(fold_m['cm'])
        results['rocs'].append(fold_m['roc'])
        results['best_cs'].append(best_C)
        results['coef'].append(model.coef_.ravel())
    return results


def run_logreg_tscv(X_all, y_all, tickers_all, features, tscv_folds):
    """LogReg L1 with C tuned per fold (last 20% of train = inner val)."""
    aucs, accs, coefs, best_cs = [], [], [], []
    for tr_idx, te_idx in tscv_folds:
        itr_idx, iv_idx = _inner_split_tscv(tr_idx)

        X_tr_s, X_te_s = preprocess_tabular(
            X_all.iloc[tr_idx], X_all.iloc[te_idx],
            tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
        X_itr_s, X_iv_s = preprocess_tabular(
            X_all.iloc[itr_idx], X_all.iloc[iv_idx],
            tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)

        best_C, _ = _tune_C(X_itr_s, y_all.iloc[itr_idx],
                             X_iv_s, y_all.iloc[iv_idx])
        m = LogisticRegression(C=best_C, **LR_DEFAULTS)
        m.fit(X_tr_s, y_all.iloc[tr_idx])
        fold_m = _evaluate_fold(m, X_te_s, y_all.iloc[te_idx])
        aucs.append(fold_m['auc'])
        accs.append(fold_m['acc'])
        coefs.append(m.coef_.ravel())
        best_cs.append(best_C)
    return dict(aucs=aucs, accs=accs, coefs=coefs, best_cs=best_cs)


# ── Per-fold PCA sweep (LogReg) — C tuned per fold ───────────────────────────

def pca_sweep_logreg(folds, X_all, y_all, tickers_all, features, k_range,
                    primary=None, folds_def=None, test_years=None, label=''):
    """Sweep k components with per-fold PCA + LogReg. C tuned per fold.

    - For year folds: pass `primary` and `folds_def` so inner val = last train year.
    - For tscv folds: omit both; inner val = last 20% of train indices.

    Returns ({k: {avg_auc, avg_acc, fold_aucs, fold_accs, best_cs}}, best_k).
    """
    is_year = folds_def is not None and primary is not None
    results = {}
    for k in k_range:
        fold_aucs, fold_accs, fold_cs = [], [], []
        for i, (tr_idx, te_idx) in enumerate(folds):
            if is_year:
                itr_idx, iv_idx = _inner_split_year(
                    primary, folds_def[i]['train_years'])
            else:
                itr_idx, iv_idx = _inner_split_tscv(tr_idx)

            X_tr_s, X_te_s = preprocess_tabular(
                X_all.iloc[tr_idx], X_all.iloc[te_idx],
                tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
            X_itr_s, X_iv_s = preprocess_tabular(
                X_all.iloc[itr_idx], X_all.iloc[iv_idx],
                tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)

            pca = PCA(n_components=k)
            X_tr_p = pca.fit_transform(X_tr_s)
            X_te_p = pca.transform(X_te_s)
            # PCA fitted on full train; reuse for inner split via same projection
            X_itr_p = pca.transform(X_itr_s)
            X_iv_p = pca.transform(X_iv_s)

            best_C, _ = _tune_C(X_itr_p, y_all.iloc[itr_idx],
                                 X_iv_p, y_all.iloc[iv_idx])
            m = LogisticRegression(C=best_C, **LR_DEFAULTS)
            m.fit(X_tr_p, y_all.iloc[tr_idx])
            fold_m = _evaluate_fold(m, X_te_p, y_all.iloc[te_idx])
            fold_aucs.append(fold_m['auc'])
            fold_accs.append(fold_m['acc'])
            fold_cs.append(best_C)

        if test_years:
            nc = [i for i, yr in enumerate(test_years) if yr != 2020]
            avg_auc = float(np.mean([fold_aucs[i] for i in nc]))
            avg_acc = float(np.mean([fold_accs[i] for i in nc]))
        else:
            avg_auc = float(np.mean(fold_aucs))
            avg_acc = float(np.mean(fold_accs))
        results[k] = dict(avg_auc=avg_auc, avg_acc=avg_acc,
                          fold_aucs=fold_aucs, fold_accs=fold_accs,
                          best_cs=fold_cs)
        if label:
            print(f'  [{label}] k={k:2d} | acc={avg_acc:.4f}  auc={avg_auc:.4f}')
    best_k = max(results, key=lambda k: results[k]['avg_auc'])
    return results, best_k


def global_pca_sweep_logreg(folds, X_all_preprocessed, y_all, k_range,
                            primary=None, folds_def=None,
                            test_years=None, label=''):
    """Global PCA sweep with per-fold C tuning."""
    is_year = folds_def is not None and primary is not None
    results = {}
    for k in k_range:
        X_pca = PCA(n_components=k).fit_transform(X_all_preprocessed)
        fold_aucs, fold_accs, fold_cs = [], [], []
        for i, (tr_idx, te_idx) in enumerate(folds):
            if is_year:
                itr_idx, iv_idx = _inner_split_year(
                    primary, folds_def[i]['train_years'])
            else:
                itr_idx, iv_idx = _inner_split_tscv(tr_idx)
            best_C, _ = _tune_C(
                X_pca[itr_idx], y_all.iloc[itr_idx],
                X_pca[iv_idx], y_all.iloc[iv_idx])
            m = LogisticRegression(C=best_C, **LR_DEFAULTS)
            m.fit(X_pca[tr_idx], y_all.iloc[tr_idx])
            fold_m = _evaluate_fold(m, X_pca[te_idx], y_all.iloc[te_idx])
            fold_aucs.append(fold_m['auc'])
            fold_accs.append(fold_m['acc'])
            fold_cs.append(best_C)
        if test_years:
            nc = [i for i, yr in enumerate(test_years) if yr != 2020]
            avg_auc = float(np.mean([fold_aucs[i] for i in nc]))
            avg_acc = float(np.mean([fold_accs[i] for i in nc]))
        else:
            avg_auc = float(np.mean(fold_aucs))
            avg_acc = float(np.mean(fold_accs))
        results[k] = dict(avg_auc=avg_auc, avg_acc=avg_acc,
                          fold_aucs=fold_aucs, fold_accs=fold_accs,
                          best_cs=fold_cs)
        if label:
            print(f'  [{label}] k={k:2d} | acc={avg_acc:.4f}  auc={avg_auc:.4f}')
    best_k = max(results, key=lambda k: results[k]['avg_auc'])
    return results, best_k


def pca_logreg_detail(folds, X_all, y_all, tickers_all, features, k,
                     primary=None, folds_def=None):
    """Full per-fold results + artifacts at chosen k with per-fold C tuning."""
    is_year = folds_def is not None and primary is not None
    out = {'test_years': [], 'aucs': [], 'accs': [], 'cms': [], 'rocs': [],
           'best_cs': [], 'feature_importances': [], 'loadings': []}
    for i, (tr_idx, te_idx) in enumerate(folds):
        if is_year:
            itr_idx, iv_idx = _inner_split_year(
                primary, folds_def[i]['train_years'])
        else:
            itr_idx, iv_idx = _inner_split_tscv(tr_idx)

        X_tr_s, X_te_s = preprocess_tabular(
            X_all.iloc[tr_idx], X_all.iloc[te_idx],
            tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
        X_itr_s, X_iv_s = preprocess_tabular(
            X_all.iloc[itr_idx], X_all.iloc[iv_idx],
            tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)

        pca = PCA(n_components=k)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_te_p = pca.transform(X_te_s)
        X_itr_p = pca.transform(X_itr_s)
        X_iv_p = pca.transform(X_iv_s)

        best_C, _ = _tune_C(X_itr_p, y_all.iloc[itr_idx],
                             X_iv_p, y_all.iloc[iv_idx])
        m = LogisticRegression(C=best_C, **LR_DEFAULTS)
        m.fit(X_tr_p, y_all.iloc[tr_idx])
        fold_m = _evaluate_fold(m, X_te_p, y_all.iloc[te_idx])

        contrib = m.coef_.ravel() @ pca.components_
        out['aucs'].append(fold_m['auc'])
        out['accs'].append(fold_m['acc'])
        out['cms'].append(fold_m['cm'])
        out['rocs'].append(fold_m['roc'])
        out['best_cs'].append(best_C)
        out['feature_importances'].append(
            dict(zip(features, np.abs(contrib))))
        out['loadings'].append(pca.components_)
        if folds_def:
            out['test_years'].append(folds_def[i]['test_year'])
        else:
            out['test_years'].append(i)
    return out


# ── XGBoost — 48-combo grid + early stopping ────────────────────────────────

def _xgb_grid_search(X_itr, y_itr, X_iv, y_iv, param_grid=XGB_PARAM_GRID):
    """Return (best_params, best_n_trees, best_auc) across grid_combinations.

    Each combination uses n_estimators=500 + early_stopping_rounds=30 on the
    inner-val set. `best_iteration + 1` is captured for the final refit.
    """
    keys = list(param_grid.keys())
    all_combos = list(product(*param_grid.values()))
    best_params, best_auc, best_n = None, -np.inf, 100
    for combo in all_combos:
        params = dict(zip(keys, combo))
        mdl = XGBClassifier(**params, **XGB_ES_DEFAULTS)
        mdl.fit(X_itr, y_itr,
                eval_set=[(X_iv, y_iv)], verbose=False)
        auc = roc_auc_score(y_iv, mdl.predict_proba(X_iv)[:, 1])
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_n = mdl.best_iteration + 1
    return best_params, max(best_n, 1), best_auc


def run_xgb_year(primary, X_all, y_all, tickers_all, features, folds_def,
                 use_pca=False, k=None, param_grid=XGB_PARAM_GRID):
    """XGBoost with 48-combo grid search on inner val (mirrors xgboost_model.ipynb)."""
    out = {'test_years': [], 'aucs': [], 'accs': [], 'cms': [], 'rocs': [],
           'feature_importances': [], 'best_params': []}
    for fold in folds_def:
        train_years = fold['train_years']
        test_year = fold['test_year']
        tr_idx = np.where(primary['Year'].isin(train_years))[0]
        te_idx = np.where(primary['Year'] == test_year)[0]
        itr_idx, iv_idx = _inner_split_year(primary, train_years)

        X_tr_s, X_te_s = preprocess_tabular(
            X_all.iloc[tr_idx], X_all.iloc[te_idx],
            tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
        X_itr_s, X_iv_s = preprocess_tabular(
            X_all.iloc[itr_idx], X_all.iloc[iv_idx],
            tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)

        if use_pca:
            pca = PCA(n_components=k).fit(X_tr_s)
            X_tr_s, X_te_s = pca.transform(X_tr_s), pca.transform(X_te_s)
            X_itr_s, X_iv_s = pca.transform(X_itr_s), pca.transform(X_iv_s)

        best_params, best_n, _ = _xgb_grid_search(
            X_itr_s, y_all.iloc[itr_idx].values,
            X_iv_s, y_all.iloc[iv_idx].values,
            param_grid=param_grid)

        xgb_f = XGBClassifier(**best_params, n_estimators=best_n,
                              random_state=42, verbosity=0)
        xgb_f.fit(X_tr_s, y_all.iloc[tr_idx].values)
        fold_m = _evaluate_fold(xgb_f, X_te_s, y_all.iloc[te_idx])

        out['test_years'].append(test_year)
        out['aucs'].append(fold_m['auc'])
        out['accs'].append(fold_m['acc'])
        out['cms'].append(fold_m['cm'])
        out['rocs'].append(fold_m['roc'])
        out['best_params'].append({**best_params, 'n_estimators': best_n})
        if use_pca:
            out['feature_importances'].append(
                {f'PC{i+1}': v for i, v in enumerate(xgb_f.feature_importances_)})
        else:
            out['feature_importances'].append(
                dict(zip(features, xgb_f.feature_importances_)))
    return out


def run_xgb_tscv(X_all, y_all, tickers_all, features, tscv_folds,
                 use_pca=False, k=None, param_grid=XGB_PARAM_GRID):
    """XGBoost over tscv folds with 48-combo grid on inner val (last 20%)."""
    out = {'aucs': [], 'accs': [], 'best_params': []}
    for tr_idx, te_idx in tscv_folds:
        itr_idx, iv_idx = _inner_split_tscv(tr_idx)

        X_tr_s, X_te_s = preprocess_tabular(
            X_all.iloc[tr_idx], X_all.iloc[te_idx],
            tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
        X_itr_s, X_iv_s = preprocess_tabular(
            X_all.iloc[itr_idx], X_all.iloc[iv_idx],
            tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)

        if use_pca:
            pca = PCA(n_components=k).fit(X_tr_s)
            X_tr_s, X_te_s = pca.transform(X_tr_s), pca.transform(X_te_s)
            X_itr_s, X_iv_s = pca.transform(X_itr_s), pca.transform(X_iv_s)

        best_params, best_n, _ = _xgb_grid_search(
            X_itr_s, y_all.iloc[itr_idx].values,
            X_iv_s, y_all.iloc[iv_idx].values,
            param_grid=param_grid)

        xgb_f = XGBClassifier(**best_params, n_estimators=best_n,
                              random_state=42, verbosity=0)
        xgb_f.fit(X_tr_s, y_all.iloc[tr_idx].values)
        fold_m = _evaluate_fold(xgb_f, X_te_s, y_all.iloc[te_idx])
        out['aucs'].append(fold_m['auc'])
        out['accs'].append(fold_m['acc'])
        out['best_params'].append({**best_params, 'n_estimators': best_n})
    return out


# ── Cross-validate the ideal n_splits for TimeSeriesSplit ──────────────────
#
# One sweep per model family — LogReg and XGBoost have different variance
# profiles across fold counts, so each family gets its own optimal n_splits.
# Both sweeps use per-fold inner-val tuning (C grid for LogReg, fixed simple
# XGBoost recipe for speed) over n_splits ∈ [6, 20]. Pick the n_splits with
# the highest mean AUC across folds.


def _sweep_row(n_splits, aucs, accs):
    return {
        'n_splits': n_splits,
        'auc_mean': float(np.mean(aucs)),
        'auc_std':  float(np.std(aucs)),
        'acc_mean': float(np.mean(accs)),
        'acc_std':  float(np.std(accs)),
    }


def sweep_tscv_n_splits_logreg(X_all, y_all, tickers_all, features,
                                n_splits_range=range(6, 21), verbose=True):
    """LogReg L1 with per-fold C tuning. Returns (DataFrame, best_n_splits)."""
    import pandas as pd
    rows = []
    for n_splits in n_splits_range:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs = [], []
        for tr_idx, te_idx in tscv.split(X_all):
            itr_idx, iv_idx = _inner_split_tscv(tr_idx)
            X_tr_s, X_te_s = preprocess_tabular(
                X_all.iloc[tr_idx], X_all.iloc[te_idx],
                tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
            X_itr_s, X_iv_s = preprocess_tabular(
                X_all.iloc[itr_idx], X_all.iloc[iv_idx],
                tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)
            best_C, _ = _tune_C(X_itr_s, y_all.iloc[itr_idx],
                                 X_iv_s, y_all.iloc[iv_idx])
            m = LogisticRegression(C=best_C, **LR_DEFAULTS)
            m.fit(X_tr_s, y_all.iloc[tr_idx])
            fold_m = _evaluate_fold(m, X_te_s, y_all.iloc[te_idx])
            aucs.append(fold_m['auc'])
            accs.append(fold_m['acc'])
        rows.append(_sweep_row(n_splits, aucs, accs))
        if verbose:
            print(f'  [LogReg] n_splits={n_splits:2d}  AUC={rows[-1]["auc_mean"]:.4f} '
                  f'± {rows[-1]["auc_std"]:.4f}  Acc={rows[-1]["acc_mean"]:.4f}')
    df = pd.DataFrame(rows)
    return df, int(df.loc[df['auc_mean'].idxmax(), 'n_splits'])


def sweep_tscv_n_splits_xgb(X_all, y_all, tickers_all, features,
                             n_splits_range=range(6, 21), verbose=True):
    """XGBoost with simple `lr=0.05 + ES=30` recipe (no 48-combo grid — too
    expensive to sweep 15 × 48 fits). Returns (DataFrame, best_n_splits)."""
    import pandas as pd
    rows = []
    for n_splits in n_splits_range:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs = [], []
        for tr_idx, te_idx in tscv.split(X_all):
            itr_idx, iv_idx = _inner_split_tscv(tr_idx)
            X_tr_s, X_te_s = preprocess_tabular(
                X_all.iloc[tr_idx], X_all.iloc[te_idx],
                tickers_all.iloc[tr_idx], tickers_all.iloc[te_idx], features)
            X_itr_s, X_iv_s = preprocess_tabular(
                X_all.iloc[itr_idx], X_all.iloc[iv_idx],
                tickers_all.iloc[itr_idx], tickers_all.iloc[iv_idx], features)
            xgb_es = XGBClassifier(**XGB_SIMPLE)
            xgb_es.fit(X_itr_s, y_all.iloc[itr_idx].values,
                       eval_set=[(X_iv_s, y_all.iloc[iv_idx].values)],
                       verbose=False)
            best_n = max(xgb_es.best_iteration + 1, 1)
            xgb_f = XGBClassifier(n_estimators=best_n, learning_rate=0.05,
                                   random_state=42, verbosity=0)
            xgb_f.fit(X_tr_s, y_all.iloc[tr_idx].values)
            fold_m = _evaluate_fold(xgb_f, X_te_s, y_all.iloc[te_idx])
            aucs.append(fold_m['auc'])
            accs.append(fold_m['acc'])
        rows.append(_sweep_row(n_splits, aucs, accs))
        if verbose:
            print(f'  [XGBoost] n_splits={n_splits:2d}  AUC={rows[-1]["auc_mean"]:.4f} '
                  f'± {rows[-1]["auc_std"]:.4f}  Acc={rows[-1]["acc_mean"]:.4f}')
    df = pd.DataFrame(rows)
    return df, int(df.loc[df['auc_mean'].idxmax(), 'n_splits'])
