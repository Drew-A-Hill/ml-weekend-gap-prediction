"""
File: utils.terminal_run_status.py
Author: Drew Hill
This file is used to iterate through and provide a UI for the dataset build such that it is easy to see progress.
"""
import random
import time
import sys
from contextlib import ExitStack
from typing import Generator
import pandas as pd
from tqdm import tqdm
import logging

logging.getLogger("yfinance").setLevel(logging.FATAL)

def eta_str(seconds: int) -> str:
    """
    Creates the string that provides an estimated time remaining until program finishes.
    :param seconds: The number of seconds until program finishes.
    :return: The string that provides an estimated time remaining until program finishes.
    """
    if seconds < 60:
        return f"{max(0, int(seconds))}s"

    else:
        return f"{max(0, round(seconds / 60), 0)} mins {seconds % 60} sec"

def next_pause_count(count: int) -> int:
    """
    Calculates the number of calls remaining until a pause for rate limiting purposes.
    :param count: The number of calls made already.
    :return: The number of calls remaining until a pause for rate limiting purposes.
    """
    remainder = count % 500
    return 500 if remainder == 0 else 500 - remainder

def ticker_iter_w_progress(desc : str, tickers: pd.Series) -> Generator[str, None]:
    """
    Creates a progress bar such that it is easier to track the progress of data pipeline.
    :param desc: The progress bar description.
    :param tickers: The series that contains all tickers.
    :yield: Each ticker.
    """
    tickers = list(tickers)
    total = len(tickers)
    start_time = time.time()

    with ExitStack() as stack:
        main_bar = stack.enter_context(
            tqdm(
                total=total,
                desc=desc,
                position=0,
                leave=True,
                file=sys.stdout,
                dynamic_ncols=False,
                ncols=80,
                bar_format="{desc}: | {bar:20} | {percentage:3.0f}% [{n_fmt} / {total_fmt}]",
            )
        )

        spacer = stack.enter_context(
            tqdm(
                total=0,
                position=1,
                leave=True,
                file=sys.stdout,
                dynamic_ncols=False,
                ncols=80,
                bar_format="{desc}",
            )
        )

        line_ticker = stack.enter_context(
            tqdm(total=0, position=2, leave=True, file=sys.stdout,
                 dynamic_ncols=False, ncols=80, bar_format="{desc}")
        )
        line_calls = stack.enter_context(
            tqdm(total=0, position=3, leave=True, file=sys.stdout,
                 dynamic_ncols=False, ncols=80, bar_format="{desc}")
        )
        line_eta = stack.enter_context(
            tqdm(total=0, position=4, leave=True, file=sys.stdout,
                 dynamic_ncols=False, ncols=80, bar_format="{desc}")
        )

        line_message = stack.enter_context(
            tqdm(total=0, position=6, leave=True, file=sys.stdout,
                 dynamic_ncols=False, ncols=80, bar_format="{desc}")
        )

        spacer.set_description_str("")

        for count, ticker in enumerate(tickers, 1):
            elapsed = time.time() - start_time
            calls_per_sec = count / elapsed if elapsed > 0 else 0.0
            eta = (total - count) / calls_per_sec if calls_per_sec > 0 else 0

            next_pause = next_pause_count(count)

            message: str = f"Next pause in: {next_pause} calls"

            line_ticker.set_description_str(f"Ticker: {ticker}")
            line_calls.set_description_str(f"Calls/sec: {calls_per_sec:.1f}")
            line_eta.set_description_str(f"Est time remaining: {eta_str(int(eta))}")
            line_message.set_description_str(f"Message: {message}")

            line_ticker.refresh()
            line_calls.refresh()
            line_eta.refresh()
            line_message.refresh()

            main_bar.update(1)

            yield ticker

            if count % 500 == 0 and count < total:
                sleep_secs = random.randint(30, 120)
                message = f"Call Count {count}:  {sleep_secs}s"

                line_message.set_description_str(f"Message: {message}")
                line_message.refresh()

                if count < 2000:
                    time.sleep(sleep_secs)
                else:
                    time.sleep(sleep_secs * 2)

                line_message.set_description_str("Message:")
                line_message.refresh()