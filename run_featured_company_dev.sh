#!/bin/bash
cd "$(dirname "$0")" || exit
export PYTHONPATH=src
python3 src/scripts/featured_companies_dev_script.py