# ML Weekend Gap Prediction

> **Project:** Weekend gap research / feature pipeline (stocks)
> **Runs locally:**  (Python + `yfinance`)
> **Primary entrypoint:** `./run_featured_company_dev.sh`

Build a small, repeatable dataset to explore **weekend gaps** (Friday close → Monday open) across a curated list of “featured” tickers — using a script-driven workflow that’s easy to run locally and iterate on.

---

## What it does

Run one command → the project:

- Loads a curated ticker list from `structured_csv_data_files/featured_companies.csv`
- Pulls market data via `yfinance`
- Builds / refreshes a local dataset for development + experimentation
- Prints progress to the terminal (with `tqdm`) while it runs

> The “featured companies” workflow is currently the main dev path. (There’s also a second dev dataset script in `src/scripts/`.)

---

## Quickstart

### 1) Clone + install dependencies

```bash
git clone https://github.com/Drew-A-Hill/ml-weekend-gap-prediction.git
cd ml-weekend-gap-prediction

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the featured ticker pipeline

```bash
chmod +x ./run_featured_company_dev.sh
./run_featured_company_dev.sh
```

To stop the run: **CTRL + C**

---

## Architecture (current)

This repo is intentionally lightweight: scripts + a `src/` package used via `PYTHONPATH`.

```
┌─────────────────────────────────────────────────────┐
│                 local python runtime                │
│                                                     │
│  run_featured_company_dev.sh                        │
│    ├─ sets PYTHONPATH=src                           │
│    └─ runs src/scripts/featured_companies_dev_script│
│                                                     │
│  Data sources: yfinance                             │
│  Data shaping: pandas                               │
│  Inputs: structured_csv_data_files/*.csv            │
└─────────────────────────────────────────────────────┘
```

---

## Project structure

```
ml-weekend-gap-prediction/
├── requirements.txt
├── run_featured_company_dev.sh
├── structured_csv_data_files/
│   └── featured_companies.csv
└── src/
    ├── __init__.py
    ├── config.py
    ├── README.md
    ├── data_io/
    ├── data_pipelines/
    ├── feature_selection/
    ├── metric_types/
    ├── models/
    ├── scripts/
    │   ├── dev_data_set_script.py
    │   └── featured_companies_dev_script.py
    └── utils/
```

---

## Data inputs

### Featured tickers

`structured_csv_data_files/featured_companies.csv` contains one column:

- `featured_tickers`

Example values include (as currently committed): `TXN`, `ADI`, `QCOM`, `MRVL`, `MPWR`, `NXPI`, etc.

---

## Scripts

Scripts live in `src/scripts/`.

| Script | What it’s for | How to run |
|-------|----------------|-----------|
| `featured_companies_dev_script.py` | Main dev workflow: run on curated tickers | `./run_featured_company_dev.sh` |
| `dev_data_set_script.py` | Secondary dev workflow / dataset builder | `python3 src/scripts/dev_data_set_script.py` *(see note below)* |

**Note:** `./run_featured_company_dev.sh` sets `PYTHONPATH=src` for you. If you run scripts directly, you may need:

```bash
export PYTHONPATH=src
python3 src/scripts/dev_data_set_script.py
```

---

## Dependencies

Pinned in `requirements.txt`:

- `pandas>=2.0`
- `yfinance>=0.2`
- `tqdm>=4.0`
- `requests>=2.0`
- `urllib3>=2.0`
- `curl-cffi>=0.7`

---

## Common issues / troubleshooting

### “ModuleNotFoundError: …” when running scripts directly
Use the provided runner script, or set:

```bash
export PYTHONPATH=src
```

### Rate limits / flaky downloads
If `yfinance` requests fail intermittently, re-run the script. (The project also depends on `requests`, `urllib3`, and `curl-cffi`, which can help with HTTP reliability depending on how the code is written.)

---

## Disclaimer

This repository is for educational/research purposes only and **not financial advice**. Market data quality/availability may vary.

---