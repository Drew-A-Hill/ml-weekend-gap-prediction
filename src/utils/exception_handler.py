"""
File: exception_handler.py
Author: Drew Hill
This file is used to manage exception handling.
"""
import sys
from collections.abc import Callable

from curl_cffi.requests.exceptions import RequestException

def run_with_exit_handling(main: Callable[[], None]) -> None:
    try:
         main()
    except KeyboardInterrupt:
        print("\n\nExecution Interrupted")
        sys.exit(0)

    except RequestException as e:
        if "Failure writing output to destination" in str(e):
            print("\n\nExecution Interrupted")
            sys.exit(0)
        raise

    except Exception:
        print("\n\nExecution Interrupted")
        sys.exit(0)