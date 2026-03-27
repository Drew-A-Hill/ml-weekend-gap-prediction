"""
build_final_dataset.py

Produces one row per (Ticker, Friday) with:
  - New momentum / drawdown / volume features computed from Tue–Fri OHLCV
  - Friday-only values for all existing technical + fundamental indicators
  - Target_Direction: 1 if next Monday Open > Friday Close, else 0

Rows without a following Monday (dataset tail) are dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'structured_csv_data_files' / 'fetched_data' / 'dataset.csv'
OUT_PATH  = ROOT / 'structured_csv_data_files' / 'aggregated' / 'final_dataset.csv'

# Friday-only pass-through features (values taken directly from the Friday row)
FRIDAY_PASSTHROUGH = [
    'RSI', 'MACD', 'ROC', 'StochPercK', 'MFI',
    'CloseVEma50', 'CloseVSma20', 'ADX', 'BollingerBandWidth', 'ATR',
    'FiveDStdDev', 'OBV', 'WeeklyReturn', 'IntraWeekVolatility',
    'WeeklyRange', 'FridayPosition', 'OpenCloseSpread', 'VolumeRatio',
    'Revenues', 'CostOfRevenues', 'GrossProfit', 'NetIncome',
    'Assets', 'Liabilities', 'Equity', 'Shares',
    'GrossMargin', 'NetMargin', 'RoA', 'RevGrowthQoQ',
]

OUTPUT_COLS = [
    'Ticker', 'Friday_Date', 'Target_Direction',
    'Four_Day_Momentum', 'Friday_Momentum',
    'Weekly_Drawdown', 'Friday_Drawdown',
    'Weekly_Bounce', 'Friday_Bounce',
    'Four_Day_Volume', 'Friday_Volume', 'Friday_Volume_Concentration',
] + FRIDAY_PASSTHROUGH


# ── Helpers ────────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['DayOfWeekNum'] = df['Date'].dt.dayofweek          # 0=Mon … 4=Fri
    iso = df['Date'].dt.isocalendar()
    df['WeekKey'] = iso['year'].astype(int) * 100 + iso['week'].astype(int)
    return df


def four_day_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (Ticker, WeekKey): max High, min Low, sum Volume across Tue–Fri rows.
    Named 'Four_Day_*' because the window is Tuesday open → Friday close.
    """
    tue_to_fri = df[df['DayOfWeekNum'].isin([1, 2, 3, 4])]
    return (
        tue_to_fri
        .groupby(['Ticker', 'WeekKey'], sort=False)
        .agg(
            Four_Day_High   = ('High',   'max'),
            Four_Day_Low    = ('Low',    'min'),
            Four_Day_Volume = ('Volume', 'sum'),
        )
        .reset_index()
    )


def tuesday_opens(df: pd.DataFrame) -> pd.DataFrame:
    """Tuesday Open per (Ticker, WeekKey) — used for Four_Day_Momentum."""
    return (
        df[df['DayOfWeekNum'] == 1][['Ticker', 'WeekKey', 'Open']]
        .rename(columns={'Open': 'Tuesday_Open'})
        .reset_index(drop=True)
    )


def monday_opens(df: pd.DataFrame) -> pd.DataFrame:
    """All Monday rows sorted for merge_asof target assignment."""
    return (
        df[df['DayOfWeekNum'] == 0][['Ticker', 'Date', 'Open']]
        .rename(columns={'Date': 'Monday_Date', 'Open': 'Monday_Open'})
        .sort_values(['Ticker', 'Monday_Date'])
        .reset_index(drop=True)
    )


def assign_target(fri: pd.DataFrame, mon: pd.DataFrame) -> pd.Series:
    """
    For each Friday row find the nearest Monday strictly ahead within 5 calendar
    days.  Returns a Series of Target_Direction (1/0) aligned to fri's index.
    NaN where no valid following Monday exists.
    """
    # merge_asof requires the key column (Date) to be globally sorted
    fri_sorted = fri[['Ticker', 'Date', 'Close']].sort_values('Date').reset_index(drop=True)
    mon_sorted  = mon.sort_values('Monday_Date').reset_index(drop=True)

    matched = pd.merge_asof(
        fri_sorted,
        mon_sorted,
        left_on='Date',
        right_on='Monday_Date',
        by='Ticker',
        direction='forward',
    )

    gap_days = (matched['Monday_Date'] - matched['Date']).dt.days
    valid = matched['Monday_Date'].notna() & (gap_days > 0) & (gap_days <= 5)

    direction = np.where(
        valid,
        (matched['Monday_Open'] > matched['Close']).astype(float),
        np.nan,
    )

    # Re-align to the original (unsorted) fri index
    result = pd.Series(direction, index=fri_sorted.index, name='Target_Direction')
    return result.reindex(fri.index)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print('Loading dataset …')
    df = load(DATA_PATH)
    n_tickers = df['Ticker'].nunique()
    print(f'  {len(df):,} rows | {n_tickers} tickers | '
          f'{df["Date"].dt.date.min()} → {df["Date"].dt.date.max()}')

    # ── Isolate Friday rows ────────────────────────────────────────────────────
    fri = df[df['DayOfWeekNum'] == 4].copy()
    print(f'  {len(fri):,} Friday rows across {fri["Date"].dt.date.nunique()} unique dates')

    # ── Merge four-day aggregates ──────────────────────────────────────────────
    fri = fri.merge(four_day_agg(df),  on=['Ticker', 'WeekKey'], how='left')
    fri = fri.merge(tuesday_opens(df), on=['Ticker', 'WeekKey'], how='left')

    # ── Compute new features ───────────────────────────────────────────────────
    fri['Four_Day_Momentum']           = (fri['Close'] - fri['Tuesday_Open'])  / fri['Tuesday_Open']
    fri['Friday_Momentum']             = (fri['Close'] - fri['Open'])           / fri['Open']
    fri['Weekly_Drawdown']             = (fri['Four_Day_High'] - fri['Close'])  / fri['Four_Day_High']
    fri['Friday_Drawdown']             = (fri['High']          - fri['Close'])  / fri['High']
    fri['Weekly_Bounce']               = (fri['Close'] - fri['Four_Day_Low'])   / fri['Four_Day_Low']
    fri['Friday_Bounce']               = (fri['Close'] - fri['Low'])            / fri['Low']
    fri['Friday_Volume']               = fri['Volume']
    fri['Friday_Volume_Concentration'] = fri['Friday_Volume'] / fri['Four_Day_Volume']

    # ── Assign target ──────────────────────────────────────────────────────────
    fri['Target_Direction'] = assign_target(fri, monday_opens(df))

    # Drop rows without a following Monday (tail of the dataset)
    before = len(fri)
    fri = fri.dropna(subset=['Target_Direction'])
    fri['Target_Direction'] = fri['Target_Direction'].astype(int)
    dropped = before - len(fri)
    print(f'  Dropped {dropped} Friday rows with no following Monday')

    # ── Finalise ───────────────────────────────────────────────────────────────
    fri['Friday_Date'] = fri['Date'].dt.date
    result = (
        fri[OUTPUT_COLS]
        .sort_values(['Ticker', 'Friday_Date'])
        .reset_index(drop=True)
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f'\nFinal dataset shape: {result.shape}')
    print(f'Date range:          {result["Friday_Date"].min()} → {result["Friday_Date"].max()}')
    print(f'Tickers:             {result["Ticker"].nunique()}')
    print(f'\nTarget distribution:')
    vc = result['Target_Direction'].value_counts().sort_index()
    print(f'  Gap down (0): {vc.get(0, 0):,}  ({vc.get(0,0)/len(result)*100:.1f}%)')
    print(f'  Gap up   (1): {vc.get(1, 0):,}  ({vc.get(1,0)/len(result)*100:.1f}%)')

    null_pct = result.isnull().mean()
    high_null = null_pct[null_pct > 0.01]
    if not high_null.empty:
        print(f'\nColumns with >1% nulls (before cleaning):')
        print(high_null.round(3).to_string())

    # ── Clean: drop Liabilities, impute remaining nulls ────────────────────────
    result = result.drop(columns=['Liabilities'])

    # Forward-fill then backward-fill per ticker (handles indicator warm-up and
    # quarterly fundamental gaps).  Any column still null after ffill/bfill
    # (e.g. a ticker with no value at all) falls back to the global column median.
    feature_cols = [c for c in result.columns if c not in ('Ticker', 'Friday_Date', 'Target_Direction')]
    result[feature_cols] = (
        result
        .groupby('Ticker')[feature_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    # Global median fallback for any remaining NaNs
    still_null = result[feature_cols].isnull().any()
    if still_null.any():
        medians = result[feature_cols].median()
        result[feature_cols] = result[feature_cols].fillna(medians)

    remaining_nulls = result.isnull().sum().sum()
    print(f'\nAfter cleaning: {remaining_nulls} nulls remaining (expect 0)')

    # ── Spot-check one row against raw data ────────────────────────────────────
    print('\n── Spot-check (CRM, first Friday) ──')
    sample = result[result['Ticker'] == 'CRM'].iloc[0]
    raw_fri = df[(df['Ticker'] == 'CRM') & (df['DayOfWeekNum'] == 4)].sort_values('Date').iloc[0]
    raw_tue = df[(df['Ticker'] == 'CRM') & (df['DayOfWeekNum'] == 1) &
                 (df['WeekKey'] == raw_fri['WeekKey'])].iloc[0]
    raw_tofri = df[(df['Ticker'] == 'CRM') & df['DayOfWeekNum'].isin([1,2,3,4]) &
                   (df['WeekKey'] == raw_fri['WeekKey'])]
    print(f"  Friday_Date:          {sample['Friday_Date']}")
    print(f"  Four_Day_Momentum:    {sample['Four_Day_Momentum']:.6f}  "
          f"(manual: {(raw_fri['Close']-raw_tue['Open'])/raw_tue['Open']:.6f})")
    print(f"  Friday_Momentum:      {sample['Friday_Momentum']:.6f}  "
          f"(manual: {(raw_fri['Close']-raw_fri['Open'])/raw_fri['Open']:.6f})")
    print(f"  Weekly_Drawdown:      {sample['Weekly_Drawdown']:.6f}  "
          f"(manual: {(raw_tofri['High'].max()-raw_fri['Close'])/raw_tofri['High'].max():.6f})")
    print(f"  Weekly_Bounce:        {sample['Weekly_Bounce']:.6f}  "
          f"(manual: {(raw_fri['Close']-raw_tofri['Low'].min())/raw_tofri['Low'].min():.6f})")
    print(f"  Four_Day_Volume:      {sample['Four_Day_Volume']:.0f}  "
          f"(manual: {raw_tofri['Volume'].sum():.0f})")
    print(f"  Fri_Vol_Conc:         {sample['Friday_Volume_Concentration']:.6f}  "
          f"(manual: {raw_fri['Volume']/raw_tofri['Volume'].sum():.6f})")
    print(f"  WeeklyReturn (pass):  {sample['WeeklyReturn']:.6f}  "
          f"(raw Fri: {raw_fri['WeeklyReturn']:.6f})")
    print(f"  Target_Direction:     {sample['Target_Direction']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f'\nSaved → {OUT_PATH}')


if __name__ == '__main__':
    main()
