"""
File: featured_companies_dev_script.py
Author: Drew Hill
This script is used to assemble the companies used for the model.
"""
import sys

import pandas as pd
from curl_cffi.requests.exceptions import RequestException

import data_io.read_write_data as rw
import data_pipelines.api_data_ingestion.data_developer as x

def main():
    companies: pd.Series = x.dev_featured_companies(
        by_industry=True,
        by_market_cap=True,
        by_profitability=True,
        by_public_age=True
    )

    rw.write_to_csv(companies, "featured_companies.csv")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExit")
        sys.exit(0)

    except RequestException as e:
        if "Failure writing output to destination" in str(e):
            print("\nExit")
            sys.exit(0)
        raise

    except Exception:
        print("\nExit")
        sys.exit(0)