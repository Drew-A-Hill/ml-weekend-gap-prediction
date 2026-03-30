"""

"""
import numpy as np
import pandas as pd

import data_io.read_write_data as rw

def agg_week(g, ticker, week_start):
    g = g.sort_values("Date")

    bow_row = g[g["Date"].dt.dayofweek == 0]
    if bow_row.empty:
        bow_row = g[g["Date"].dt.dayofweek == 1]
    bow = bow_row.iloc[0] if not bow_row.empty else g.iloc[0]

    eow_row = g[g["Date"].dt.dayofweek == 4]
    if eow_row.empty:
        eow_row = g[g["Date"].dt.dayofweek == 3]
    eow = eow_row.iloc[-1] if not eow_row.empty else g.iloc[-1]

    first = g.iloc[0]
    last = g.iloc[-1]

    weekly_return = (last["Close"] - first["Open"]) / first["Open"] if first["Open"] != 0 else np.nan
    weekly_high = g["High"].max()
    weekly_low = g["Low"].min()
    weekly_range = (weekly_high - weekly_low) / first["Open"] if first["Open"] != 0 else np.nan

    return pd.Series({
        "WeekStart": week_start.date(),
        "WeekEnd": eow["Date"].date(),
        "Ticker": ticker,
        "Year": first["Year"],
        "Quarter": first["Quarter"],
        "Week": week_start.isocalendar()[1],
        "BoWOpen": bow["Open"],
        "BoWClose": bow["Close"],
        "EoWOpen": eow["Open"],
        "EoWClose": eow["Close"],
        "AvgVolume": g["Volume"].mean(),
        "Revenues": first["Revenues"],
        "CostOfRevenues": first["CostOfRevenues"],
        "GrossProfit": first["GrossProfit"],
        "NetIncome": first["NetIncome"],
        "Assets": first["Assets"],
        "Liabilities": first["Liabilities"],
        "Equity": first["Equity"],
        "Shares": first["Shares"],
        "weekly_return": weekly_return,
        "weekly_high": weekly_high,
        "weekly_low": weekly_low,
        "weekly_range": weekly_range,
        "GrossMargin": first["GrossMargin"],
        "NetMargin": first["NetMargin"],
        "RoA": first["RoA"],
        "RevGrowthQoQ": first["RevGrowthQoQ"],
        # "VolumeRatio": g["VolumeRatio"].mean(),
        "RSI": eow["RSI"],
        "MACD": eow["MACD"],
        "ADX": eow["ADX"],
        "ATR": eow["ATR"],
        "BollingerBandWidth": eow["BollingerBandWidth"],
        # "OBV": eow["OBV"],
        # "MFI": eow["MFI"],
    })

def weekly_observations_dataset():
    """

    """
    df: pd.DataFrame = rw.read_from_csv("datasets/base_dataset_all.csv")
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("America/New_York")
    df["Date"] = df["Date"].dt.normalize()
    df["WeekStart"] = df["Date"] - pd.to_timedelta(df["Date"].dt.dayofweek, unit="D")

    weekly_df = (
        df.groupby(["Ticker", "WeekStart"], group_keys=False)
        .apply(lambda g: agg_week(g, g.name[0], g.name[1]), include_groups=False)
        .dropna(subset=["BoWOpen", "EoWClose"])
        .reset_index(drop=True)
    )

    rw.write_to_csv(weekly_df, "datasets/dataset_weekly.csv")

if __name__ == '__main__':
    weekly_observations_dataset()