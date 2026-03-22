"""
File: exception_handler.py
Author: Drew Hill
This file is used to manage exception handling.
"""
import logging
import sys
from collections.abc import Callable
from curl_cffi.requests.exceptions import RequestException

logging.getLogger("curl_cffi").setLevel(logging.FATAL)

def run_with_exit_handling(main: Callable[[], None]) -> None:
    try:
         main()

    except KeyboardInterrupt:
        # print("\n\nExit Message: User exited the script execution.")
        sys.exit(0)

    except RequestException as e:
        if "Failure writing output to destination" in str(e):
            print("\n\nError")
            sys.exit(0)
        raise

    except Exception as e:
        print("Unexpected error: {e}".format(e=e))
        # sys.exit(0)
        raise
