"""
Build `Weekend_dataset.csv`: rows from `dataset.csv` whose session dates are
Monday or Friday only (US equity calendar; weekday from the Date column).

Run from repo root:
  PYTHONPATH=src python3 -m scripts.build_weekend_dataset_script
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import DATA_DIR
from data_io.read_write_data import read_from_csv, write_to_csv


def build_weekend_dataset(
    input_rel: str = "fetched_data/dataset.csv",
    output_rel: str = "fetched_data/Weekend_dataset.csv",
) -> pd.DataFrame:
    """
    Filter full daily panel to Monday and Friday rows; add Weekday label column.

    :param input_rel: Path under structured_csv_data_files/ to dataset.csv
    :param output_rel: Path under structured_csv_data_files/ for output CSV
    :returns: The filtered dataframe (also written to CSV)
    """
    df = read_from_csv(input_rel)
    if "Date" not in df.columns:
        raise ValueError("dataset must contain a Date column")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert(None)

    wd = df["Date"].dt.weekday
    out = df[wd.isin([0, 4])].copy()
    out["Weekday"] = out["Date"].dt.day_name()

    sort_cols = [c for c in ("Ticker", "Date") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort")

    write_to_csv(out, output_rel)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mon/Fri-only weekend dataset CSV.")
    parser.add_argument(
        "--input",
        default="fetched_data/dataset.csv",
        help="Input CSV path relative to structured_csv_data_files/",
    )
    parser.add_argument(
        "--output",
        default="fetched_data/Weekend_dataset.csv",
        help="Output CSV path relative to structured_csv_data_files/",
    )
    args = parser.parse_args()

    n = len(build_weekend_dataset(input_rel=args.input, output_rel=args.output))
    print(f"Wrote {n} rows to {DATA_DIR / args.output}")


if __name__ == "__main__":
    main()

