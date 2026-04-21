# ML Weekend Gap Prediction

> **Project:** Predicting weekend price gaps in US software equities using machine learning
> **Scope:** 25 large-cap software companies, 2016–2024, weekly observations
> **Runs locally:** Python + `yfinance` + SEC XBRL

Build a dataset and train three ML models (Logistic Regression, XGBoost, LSTM) to predict whether Monday's opening price will be higher or lower than the previous Friday's closing price.

---

## Overview

This project investigates whether publicly available technical indicators (momentum, trend, volatility, volume) and fundamental metrics can predict the direction of weekend price gaps. The pipeline pulls daily price data from Yahoo Finance and quarterly financial filings from SEC XBRL, engineers a set of weekly features, and evaluates three classes of models under walk-forward validation.

Target variable (binary): `GapUp = 1` if Monday Open > previous Friday Close, else `0`.

---

## Quickstart

### 1) Clone and install dependencies

```bash
git clone https://github.com/Drew-A-Hill/ml-weekend-gap-prediction.git
cd ml-weekend-gap-prediction

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the data pipeline

```bash
chmod +x ./run_script.sh
./run_script.sh
```

This presents an interactive menu to run one of three scripts:
1. `collect_company_metadata` — fetch SEC tickers and yfinance metadata
2. `build_list_of_companies` — apply filters from `config.py`
3. `dev_data_set` — build the full feature dataset

### 3) Explore the notebooks

```bash
jupyter notebook notebooks/
```

Notebooks in `notebooks/` cover EDA, statistical analysis, and the three model implementations.

To stop any run: **CTRL + C**

---

## Project structure

```
ml-weekend-gap-prediction/
├── requirements.txt
├── run_script.sh
├── notebooks/                    # EDA, statistical analysis, model notebooks
├── structured_csv_data_files/    # CSV inputs and pipeline outputs
└── src/
    ├── config.py
    ├── data_io/
    ├── data_pipelines/           # company selection, data retrieval, indicators
    ├── scripts/
    └── utils/
```

---

## Libraries used

**Data collection and processing**
- `pandas` — dataset construction and manipulation
- `numpy` — numerical operations
- `yfinance` — daily OHLCV price data
- `requests`, `urllib3`, `curl-cffi` — SEC XBRL API calls
- `tqdm` — progress bars

**Statistical analysis**
- `scipy` — Shapiro-Wilk, Spearman, Mann-Whitney tests
- `statsmodels` — ADF stationarity test, ACF/PACF, VIF, Logit for AIC/BIC

**Machine learning**
- `scikit-learn` — logistic regression, preprocessing (StandardScaler, etc.), metrics
- `xgboost` — gradient boosted trees
- `torch` — LSTM implementation

**Visualisation**
- `matplotlib` — plots and charts
- `seaborn` — correlation heatmaps, styled tables

> Note: `requirements.txt` currently lists only data-collection dependencies. Install `scikit-learn`, `xgboost`, `torch`, `statsmodels`, `scipy`, `matplotlib`, and `seaborn` separately if running the notebooks.

---

## Common issues / troubleshooting

### "ModuleNotFoundError: ..." when running scripts directly

Use the provided runner script, or set:

```bash
export PYTHONPATH=src
```

### Rate limits / flaky downloads

If `yfinance` or SEC requests fail intermittently, re-run the script. The pipeline includes built-in rate limiting for the SEC API — do not remove the pauses in `utils/terminal_run_status.py`.

### XGBoost `libxgboost.dylib` not loaded (macOS)

Install OpenMP runtime:

```bash
brew install libomp
```

---

## Disclaimer

This repository is for educational and research purposes only and **not financial advice**. Market data quality and availability may vary.
