"""
File: utils.terminal_run_status.py
Author: Drew Hill
This file is used to iterate through provided tickers and outputs a UI for the progress. Performs no action. Is only an
iterator with ui
"""
import random
import sys
import time
from contextlib import ExitStack
from typing import Generator
from urllib.error import HTTPError
import pandas as pd
from tqdm import tqdm
import logging

logging.getLogger("yfinance").setLevel(logging.FATAL)
logging.getLogger("curl_cffi").setLevel(logging.FATAL)

_TOTAL_SLEEP_TIME: int = 0

def _set_up_main_bar(stack: ExitStack, total: int, desc: str) -> tqdm:
    """
    Sets up the main progress bar.
    :param stack: The exits stack.
    :param total: The total number of calls to be made.
    :return: A tqdm object.
    """
    return stack.enter_context(
            tqdm(
                total=total,
                desc=desc,
                position=0,
                leave=False,
                dynamic_ncols=False,
                ncols=80,
                bar_format="{desc}: | {bar:20} | {percentage:3.0f}% [{n_fmt} / {total_fmt}]",
            )
        )

def _set_up_spacer(stack: ExitStack) -> tqdm:
    """
    Sets up the spacer for the progress bar.
    :param stack: The exits stack.
    :return: A tqdm object.
    """
    return stack.enter_context(
        tqdm(
            total=0,
            position=1,
            leave=True,
            dynamic_ncols=False,
            ncols=80,
            bar_format="{desc}",
        )
    )

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

def _pause_count_down(pause: int, line_message: tqdm) -> None:
    """

    :return:
    """
    global _TOTAL_SLEEP_TIME
    _TOTAL_SLEEP_TIME = _TOTAL_SLEEP_TIME + pause

    remaining_secs = pause

    for _ in range(pause):
        minutes = remaining_secs // 60
        seconds = remaining_secs % 60

        line_message.set_description_str(f"Message: Rate limit pause | resumes in {minutes}:{seconds:02d}")

        time.sleep(1)
        remaining_secs -= 1

def _pause_calls(line_message: tqdm, count: int, total: int) -> None:
    """
    Runs the rate limit pauses based on number of calls made.
    :param line_message:
    :param count:
    :param total:
    :return:
    """
    try:
        if count % 500 == 0 and count < total:
            sleep_secs = random.randint(30, 120)

            if count < 2000:
                _pause_count_down(sleep_secs, line_message)
            else:
                _pause_count_down(sleep_secs * 2, line_message)

            line_message.set_description_str("~ ")

    except KeyboardInterrupt:
        line_message.set_description_str("Exit Message: User exited the script execution.")
        sys.exit(0)

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
        main_bar = _set_up_main_bar(stack, total, desc)
        spacer = _set_up_spacer(stack)

        line_ticker = stack.enter_context(
            tqdm(total=0, position=2, leave=True, dynamic_ncols=False, ncols=80, bar_format="{desc}"))

        line_calls = stack.enter_context(
            tqdm(total=0, position=3, leave=True, dynamic_ncols=False, ncols=80, bar_format="{desc}"))

        line_eta = stack.enter_context(
            tqdm(total=0, position=4, leave=True, dynamic_ncols=False, ncols=80, bar_format="{desc}"))

        space_1 = stack.enter_context(
            tqdm(total=0, position=5, leave=True, dynamic_ncols=False, ncols=80, bar_format="{desc}"))

        line_message = stack.enter_context(
            tqdm(total=0, position=6, leave=True, dynamic_ncols=False, ncols=80, bar_format="{desc}"))

        spacer.set_description_str("")

        try:
            for count, ticker in enumerate(tickers, 1):
                elapsed = (time.time() - start_time) - _TOTAL_SLEEP_TIME
                calls_per_sec = count / elapsed if elapsed > 0 else 0.0
                eta = (total - count) / calls_per_sec if calls_per_sec > 0 else 0

                line_ticker.set_description_str(f"Ticker: {ticker}")
                line_calls.set_description_str(f"Calls/sec: {calls_per_sec:.1f}")
                line_eta.set_description_str(f"Est time remaining: {eta_str(int(eta))}")
                space_1.set_description_str("")
                line_message.set_description_str(f"~ ")

                main_bar.update(1)

                yield ticker

                _pause_calls(line_message, count, total)

        except KeyboardInterrupt:
            raise

        except HTTPError:
            line_message.set_description_str(f"Message: {ticker} Not Found")

        line_message.set_description_str(f"Message: Execution Complete")