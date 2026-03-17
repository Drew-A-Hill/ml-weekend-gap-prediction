"""
File: dev_data_set_script.py
Author: Drew Hill
This file runs the date set builder.
"""
import sys

import pandas as pd
from curl_cffi.requests.exceptions import RequestException

import data_io.read_write_data as rw
from data_pipelines.api_data_ingestion.data_developer import dev_price_and_fundamental_data_by_ticker

def  main():
    companies: pd.DataFrame = rw.read_from_csv("featured_companies.csv")
    print(dev_price_and_fundamental_data_by_ticker(companies))

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