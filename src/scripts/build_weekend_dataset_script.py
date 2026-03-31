"""
Build Weekend_dataset.csv from fetched_data/dataset.csv.

Keeps only Monday and Friday rows (based on the parsed Date field).
Output: structured_csv_data_files/Weekend_dataset.csv
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path


def _repo_root() -> Path:
    # src/scripts/<this_file> -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _weekday_from_date_str(s: str) -> int | None:
    """
    Returns weekday where Monday=0 ... Sunday=6, or None if parse fails.

    dataset.csv values look like: "2016-03-28 00:00:00-04:00"
    """
    if not s:
        return None
    try:
        # Python's fromisoformat supports "YYYY-MM-DD HH:MM:SS±HH:MM"
        return dt.datetime.fromisoformat(s).weekday()
    except ValueError:
        return None


def build_weekend_dataset() -> Path:
    root = _repo_root()
    input_csv = root / "structured_csv_data_files" / "fetched_data" / "dataset.csv"
    output_csv = root / "structured_csv_data_files" / "Weekend_dataset.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_csv}")

    rows_in = 0
    rows_out = 0

    with input_csv.open("r", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row")
        if "Date" not in reader.fieldnames:
            raise ValueError("Expected column 'Date' in dataset.csv")

        with output_csv.open("w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                rows_in += 1
                wd = _weekday_from_date_str(row.get("Date", ""))
                if wd in (0, 4):  # Monday or Friday
                    writer.writerow(row)
                    rows_out += 1

    print(f"Read {rows_in} rows, wrote {rows_out} rows")
    return output_csv


def main() -> None:
    out = build_weekend_dataset()
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

