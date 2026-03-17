#!/bin/bash
cd "$(dirname "$0")" || exit
export PYTHONPATH=src
python3 src/scripts/script_runner.py