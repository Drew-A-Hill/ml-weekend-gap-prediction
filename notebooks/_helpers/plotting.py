"""Reusable matplotlib helpers — ROC grids, confusion matrices, AUC lines,
feature importance bars, tscv histograms. Kept dependency-light (no seaborn)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_roc_grid(rocs, aucs, titles, suptitle='', ncols=3, covid_year=None):
    """2-row grid of ROC curves, one per fold."""
    n = len(rocs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, ((fpr, tpr), auc, title) in enumerate(zip(rocs, aucs, titles)):
        ax = axes[i]
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC={auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f'Test {title}', fontsize=11)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower right', fontsize=9)
        if covid_year and title == covid_year:
            ax.set_facecolor('#fffbe6')
    for ax in axes[n:]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_cm_grid(cms, titles, suptitle='', labels=('Gap Down', 'Gap Up'),
                 ncols=3, covid_year=None):
    """Grid of confusion matrices."""
    n = len(cms)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, (cm, title) in enumerate(zip(cms, titles)):
        ax = axes[i]
        ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels([f'True {labels[0]}', f'True {labels[1]}'])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                        fontsize=13,
                        color='white' if cm[r, c] > cm.max() / 2 else 'black')
        ax.set_title(f'Test {title}', fontsize=11)
        if covid_year and title == covid_year:
            for spine in ax.spines.values():
                spine.set_edgecolor('#ffc107')
                spine.set_linewidth(2.0)
    for ax in axes[n:]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_auc_per_fold(test_years, series_list, title='', ylim=(0.48, 0.65),
                      covid_year=2020):
    """series_list: list of (label, values, color, marker, linestyle)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, values, color, marker, ls in series_list:
        ax.plot(test_years, values, marker=marker, ls=ls, color=color,
                lw=2, ms=7, label=label)
    ax.axhline(0.5, color='grey', ls=':', lw=1, label='Random')
    if covid_year and covid_year in test_years:
        ax.axvspan(covid_year - 0.5, covid_year + 0.5,
                   color='#fff3cd', alpha=0.6, label='COVID fold')
    ax.set_xlabel('Test Year')
    ax.set_ylabel('AUC-ROC')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(imp_df, title='', top_n=None):
    """Horizontal bar chart of average feature importance (AUC drop)."""
    df = imp_df.copy()
    if top_n:
        df = df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.33 * len(df))))
    colors = ['steelblue' if v >= 0 else 'tomato' for v in df['Avg AUC drop']]
    ax.barh(df.index, df['Avg AUC drop'], color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Average AUC drop (higher = more important)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_importance_heatmap(imp_df, title=''):
    """Stability heatmap — importance per fold, features × test_years."""
    data = imp_df.drop(columns='Avg AUC drop').astype(float)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(data))))
    im = ax.imshow(data.values, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    plt.colorbar(im, ax=ax, label='AUC drop')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_tscv_distribution(lr_aucs, xgb_aucs, lr_accs, xgb_accs, n_folds=18):
    """Twin-panel histograms of AUC/Accuracy for two models across tscv folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'tscv({n_folds}) Per-Fold Distribution — LogReg L1 vs XGBoost',
                 fontsize=14, fontweight='bold')
    bins = np.arange(0.45, 0.70, 0.02)
    for ax, metric, lr_v, xgb_v in [
        (axes[0], 'AUC-ROC', lr_aucs, xgb_aucs),
        (axes[1], 'Accuracy', lr_accs, xgb_accs),
    ]:
        ax.hist(lr_v, bins=bins, alpha=0.65, color='steelblue',
                edgecolor='white', linewidth=0.8, label='LogReg L1')
        ax.hist(xgb_v, bins=bins, alpha=0.65, color='darkorange',
                edgecolor='white', linewidth=0.8, label='XGBoost')
        ax.axvline(float(np.mean(lr_v)), color='steelblue', ls='--', lw=2,
                   label=f'LogReg mean={np.mean(lr_v):.4f}')
        ax.axvline(float(np.mean(xgb_v)), color='darkorange', ls='--', lw=2,
                   label=f'XGBoost mean={np.mean(xgb_v):.4f}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Number of folds')
        ax.set_title(f'{metric} distribution (n={n_folds} folds)')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.show()
