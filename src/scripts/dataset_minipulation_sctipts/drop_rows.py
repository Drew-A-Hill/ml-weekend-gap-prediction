"""

"""
import pandas as pd

from data_io.read_write_data import read_from_csv, write_to_csv

def remove_extra_years(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["Year"].isin([2013, 2025, 2026])]

def remove_companies(df: pd.DataFrame) -> pd.DataFrame:
    tickers = ["DSK", "AKAM", "CDNS", "INTU", "WDAY", "FFIV"]
    return df[~df["Ticker"].isin(tickers)]

def main():
    df = read_from_csv("datasets/dataset_weekly.csv")
    df = remove_extra_years(df)
    df = remove_companies(df)
    write_to_csv(df, "datasets/dataset_weekly_cleaned.csv")

if __name__ == '__main__':
    main()