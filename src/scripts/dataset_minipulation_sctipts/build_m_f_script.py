"""
Filename: build_m_f_script.py
Authors: Drew Hill
This script removes days that are not monday or friday from the observations.
"""
import pandas as pd
import data_io.read_write_data as rw

def build_weekly():
    df: pd.DataFrame = rw.read_from_csv("fetched_data/permanent/dataset2.csv")

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df[df["Date"].dt.dayofweek.isin([0, 4])]

    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    first_cols = ["Date", "Year", "Quarter", "WeekOfYear", "DayOfWeek", "Ticker"]
    remaining = [title for title in df.columns if title not in first_cols]
    df = df[first_cols + remaining]

    rw.write_to_csv(df, "fetched_data/dataset_mon_fri.csv")

if __name__ == '__main__':
    build_weekly()
