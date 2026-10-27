#!/bin/sh
set -e
cp "$1/tests/test_stats.py" ./test_stats_hidden.py
python3 -m pytest -q test_stats_hidden.py
