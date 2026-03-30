"""
File: script_runner.py
Author: Drew Hill
This file is used to run the desired scripts with exit handling.
"""
from collections.abc import Callable

import scripts.data_collection_scripts.build_filtered_company_list_script as fc
import scripts.data_collection_scripts.build_dataset_script as ds
import scripts.data_collection_scripts.collect_metadata_script as md

from utils.exception_handler import run_with_exit_handling

SCRIPT_MAP: dict[str, Callable[[], None]] = {
    "collect_company_metadata": md.collect_companies_meta_data,
    "build_list_of_companies": fc.build_filtered_list,
    "dev_data_set": ds.build_base_dataset_price_fundamentals
}

def _callable_script_list() -> str:
    """

    :return:
    """
    callable_scripts: str = ""

    count: int = 1

    for script in SCRIPT_MAP.keys():
        callable_scripts += f"{count}: {script}\n"
        count += 1

    return callable_scripts

def _execution_message(script_to_run: str) -> None:
    """
    Prints execution message to console.
    :param script_to_run: The script being executed.
    """
    print("-" * 100)
    print(f"\nRunning {script_to_run}\n")
    print("-" * 100)

def main():
    """

    :return:
    """
    attempts: int = 0
    script_to_run: str = ""

    while not script_to_run:
        question: str = f"Which script would you like to run?\n{_callable_script_list()} \n ~ Run: "

        response_in: str = input(question).strip().lower()

        try:
            script: str = list(SCRIPT_MAP.keys())[int(response_in) - 1]

        except IndexError:
            print(f"Select a a value between 1 and {len(SCRIPT_MAP.keys())}")
            attempts += 1
            continue

        if script in SCRIPT_MAP.keys():
            script_to_run = script

        elif attempts > 5:
            attempts += 1

        else:
            raise Exception("Try again later")

    _execution_message(script_to_run)

    run_with_exit_handling(SCRIPT_MAP[script_to_run])

if __name__ == '__main__':
    main()