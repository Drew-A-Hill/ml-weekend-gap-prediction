"""
Final model comparison chart — AUC-ROC and Accuracy by test year.

Sources:
  LogReg L1           — logistic_regression.ipynb (canonical)
  LogReg + PCA        — PCA.ipynb / pca_results.pkl  (pca_logreg, year folds, k=17)
  XGBoost             — xgboost_model.ipynb (canonical, 21 features)
  LSTM baseline       — lstm_model.ipynb (StandardScaler, 14 features)
  LSTM LagCC          — lstm_std_results.pkl (StandardScaler, 11 LagCC features)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Canonical per-fold data ────────────────────────────────────────────────────
NC_YEARS = [2019, 2021, 2022, 2023, 2024]

# LogReg L1 — logistic_regression.ipynb
LOGREG = {
    'auc': {2019: 0.597, 2020: 0.563, 2021: 0.528, 2022: 0.595, 2023: 0.570, 2024: 0.579},
    'acc': {2019: 0.560, 2020: 0.574, 2021: 0.514, 2022: 0.571, 2023: 0.553, 2024: 0.564},
}

# XGBoost — xgboost_model.ipynb
XGBOOST = {
    'auc': {2019: 0.550, 2020: 0.564, 2021: 0.534, 2022: 0.593, 2023: 0.569, 2024: 0.586},
    'acc': {2019: 0.558, 2020: 0.550, 2021: 0.524, 2022: 0.568, 2023: 0.546, 2024: 0.575},
}

# LSTM baseline — lstm_model.ipynb
LSTM_BASE = {
    'auc': {2019: 0.565, 2020: 0.561, 2021: 0.548, 2022: 0.575, 2023: 0.554, 2024: 0.544},
    'acc': {2019: 0.552, 2020: 0.556, 2021: 0.538, 2022: 0.520, 2023: 0.547, 2024: 0.506},
}

# LogReg + PCA and LSTM LagCC — from pickles
with open('structured_csv_data_files/pca_results.pkl', 'rb') as f:
    pca_d = pickle.load(f)

pca_lr = pca_d['pca_logreg']
LOGREG_PCA = {
    'auc': dict(zip(pca_lr['years'], pca_lr['aucs'])),
    'acc': dict(zip(pca_lr['years'], pca_lr['accs'])),
}

with open('structured_csv_data_files/lstm_std_results.pkl', 'rb') as f:
    std_d = pickle.load(f)

lagcc_df = std_d['lagcc_roll']
LSTM_LAGCC = {
    'auc': {int(yr): row['AUC-ROC']  for yr, row in lagcc_df.iterrows()},
    'acc': {int(yr): row['Accuracy'] for yr, row in lagcc_df.iterrows()},
}

# ── Verify numbers ─────────────────────────────────────────────────────────────
models = [
    ('LogReg L1',      LOGREG,      '#f28e2b'),
    ('LogReg + PCA',   LOGREG_PCA,  '#ffbe7d'),
    ('XGBoost',        XGBOOST,     '#59a14f'),
    ('LSTM baseline',  LSTM_BASE,   '#9c755f'),
    ('LSTM LagCC',     LSTM_LAGCC,  '#4e79a7'),
]

SOURCES = {
    'LogReg L1':     'logistic_regression.ipynb',
    'LogReg + PCA':  'PCA.ipynb → pca_results.pkl [pca_logreg]',
    'XGBoost':       'xgboost_model.ipynb (21 features)',
    'LSTM baseline': 'lstm_model.ipynb (14 feat, StandardScaler)',
    'LSTM LagCC':    'lstm_std_results.pkl [lagcc_roll] (11 feat, StandardScaler)',
}

print('Non-COVID per-year AUC & Accuracy')
print('%-18s' % 'Model' + ''.join(['   %4d  ' % y for y in NC_YEARS]) + '  Avg')
print('─' * 80)
for label, d, _ in models:
    aucs = [d['auc'][y] for y in NC_YEARS]
    accs = [d['acc'][y] for y in NC_YEARS]
    print('%-18s AUC  %s  %.4f' % (
        label, '  '.join(['%.3f' % a for a in aucs]), np.mean(aucs)))
    print('%-18s Acc  %s  %.4f' % (
        '',    '  '.join(['%.3f' % a for a in accs]), np.mean(accs)))
    print()

# ── Chart ──────────────────────────────────────────────────────────────────────
n_m     = len(models)
x       = np.arange(len(NC_YEARS))
width   = 0.15
offsets = np.linspace(-(n_m - 1) / 2 * width, (n_m - 1) / 2 * width, n_m)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Comparison by Test Year — non-COVID folds',
             fontsize=14, fontweight='bold', y=1.01)

for ax, metric, ylim, ylabel in [
    (axes[0], 'auc', (0.48, 0.64), 'AUC-ROC'),
    (axes[1], 'acc', (0.48, 0.62), 'Accuracy'),
]:
    for (label, d, color), off in zip(models, offsets):
        vals = [d[metric][y] for y in NC_YEARS]
        ax.bar(x + off, vals, width, label=label, color=color,
               alpha=0.9, edgecolor='white', linewidth=0.7)

    ax.axhline(0.5, color='grey', linestyle=':', linewidth=1.2, label='Random (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in NC_YEARS], fontsize=11)
    ax.set_xlabel('Test Year', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ylabel, fontsize=13, fontweight='bold')
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].legend(fontsize=10, loc='lower right')

source_lines = '\n'.join(
    f'{label}: {SOURCES[label]}' for label, _, _ in models
)
fig.text(0.01, -0.04, 'Sources:\n' + source_lines,
         fontsize=7.5, va='top', family='monospace',
         color='#444444')

plt.tight_layout()
out = 'structured_csv_data_files/final_comparison_chart.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print('Saved:', out)
