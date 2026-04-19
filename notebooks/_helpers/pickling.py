"""Shared result-pickling schema + helpers.

Each notebook (PCA, LSTM) pickles one file in `final_outputs/` containing the
family's best-performing variant plus a summary of every variant tested. The
final_comparison notebook consumes these to produce cross-family charts.
"""
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

FINAL_OUTPUTS_DIR = '../final_outputs'


def save_results(path: str, payload: dict) -> str:
    """Pickle `payload` to an absolute/relative path; ensures parent exists."""
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, 'wb') as f:
        pickle.dump(payload, f)
    return abs_path


def load_results(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_result_payload(model_family: str,
                         best_variant: dict,
                         all_variants: dict,
                         meta: dict | None = None) -> dict:
    """Standard payload schema shared by PCA and LSTM notebooks.

    Parameters
    ----------
    model_family : 'PCA' | 'LSTM'
    best_variant : dict with keys:
        name (str), selection (str), hyperparams (dict), n_features (int),
        yearly (dict), tscv (dict), feature_importance (dict or None),
        confusion_matrices (list), rocs (list), test_years (list)
    all_variants : dict[str, dict] — summary per variant
        { name: {'yearly_auc': float, 'yearly_acc': float,
                 'tscv_auc': float, 'tscv_acc': float,
                 'n_features': int, ...} }
    meta : dict — optional free-form context (dataset size, dates, notes)
    """
    return {
        'model_family': model_family,
        'best_variant': best_variant,
        'all_variants': all_variants,
        'meta': meta or {},
    }


def summarize_variant(name: str,
                      yearly_aucs: list[float], yearly_accs: list[float],
                      yearly_test_years: list[int],
                      tscv_aucs: list[float] | None = None,
                      tscv_accs: list[float] | None = None,
                      n_features: int | None = None,
                      extra: dict | None = None) -> dict:
    """Build one variant's summary row (non-COVID avg for year folds)."""
    nc = [i for i, yr in enumerate(yearly_test_years) if yr != 2020]
    summary = {
        'name': name,
        'n_features': n_features,
        'yearly_test_years': list(yearly_test_years),
        'yearly_aucs': list(yearly_aucs),
        'yearly_accs': list(yearly_accs),
        'yearly_auc_nc': float(np.mean([yearly_aucs[i] for i in nc])) if nc else np.nan,
        'yearly_acc_nc': float(np.mean([yearly_accs[i] for i in nc])) if nc else np.nan,
        'tscv_aucs': list(tscv_aucs) if tscv_aucs is not None else None,
        'tscv_accs': list(tscv_accs) if tscv_accs is not None else None,
        'tscv_auc': (float(np.mean(tscv_aucs))
                     if tscv_aucs is not None else None),
        'tscv_acc': (float(np.mean(tscv_accs))
                     if tscv_accs is not None else None),
    }
    if extra:
        summary.update(extra)
    return summary


def pick_best(all_variants: dict, criterion: str = 'yearly_auc_nc') -> tuple[str, dict]:
    """Pick the variant with the highest `criterion` (default: non-COVID yearly AUC).

    Returns (variant_name, variant_summary).
    """
    def score(v):
        s = v.get(criterion)
        return -np.inf if s is None or np.isnan(s) else s
    name = max(all_variants, key=lambda k: score(all_variants[k]))
    return name, all_variants[name]
