"""
EDA helpers for local fetched CSV outputs.

Reads:
- structured_csv_data_files/fetched_data/dataset.csv
- structured_csv_data_files/fetched_data/filtered_company_list.csv

and prints lightweight summaries to stdout (no extra deps beyond pandas).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EdaPaths:
    dataset_csv: Path
    filtered_company_list_csv: Path


def _repo_root() -> Path:
    # src/scripts/<this_file> -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def default_paths() -> EdaPaths:
    root = _repo_root()
    fetched = root / "structured_csv_data_files" / "fetched_data"
    return EdaPaths(
        dataset_csv=fetched / "dataset.csv",
        filtered_company_list_csv=fetched / "filtered_company_list.csv",
    )


def _print_df_overview(df: pd.DataFrame, *, name: str, head_n: int = 5) -> None:
    print("=" * 100)
    print(f"{name}: shape={df.shape}")
    print("-" * 100)
    print("dtypes:")
    print(df.dtypes.sort_index())
    print("-" * 100)
    na = df.isna().mean().sort_values(ascending=False)
    top_na = na.head(20)
    print("top missingness (fraction):")
    print(top_na.to_string())
    print("-" * 100)
    print(f"head({head_n}):")
    print(df.head(head_n).to_string(index=False))


def eda_dataset(dataset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)

    # Best-effort parse of Date column if present (keeps original string if parsing fails)
    if "Date" in df.columns:
        df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)

    _print_df_overview(df, name="dataset.csv")

    if "Ticker" in df.columns:
        print("-" * 100)
        print("rows per Ticker (top 20):")
        print(df["Ticker"].value_counts().head(20).to_string())

    if {"Ticker", "Year", "Quarter"}.issubset(df.columns):
        print("-" * 100)
        print("unique (Ticker, Year, Quarter) rows:")
        print(df[["Ticker", "Year", "Quarter"]].drop_duplicates().shape[0])

    return df


def eda_filtered_company_list(filtered_company_list_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(filtered_company_list_csv)
    _print_df_overview(df, name="filtered_company_list.csv")

    if "Sector" in df.columns:
        print("-" * 100)
        print("count by Sector:")
        print(df["Sector"].value_counts().to_string())

    if "Industry" in df.columns:
        print("-" * 100)
        print("count by Industry (top 25):")
        print(df["Industry"].value_counts().head(25).to_string())

    return df


def main() -> None:
    paths = default_paths()

    if not paths.dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {paths.dataset_csv}")
    if not paths.filtered_company_list_csv.exists():
        raise FileNotFoundError(f"Filtered company list CSV not found: {paths.filtered_company_list_csv}")

    eda_dataset(paths.dataset_csv)
    eda_filtered_company_list(paths.filtered_company_list_csv)


if __name__ == "__main__":
    main()

