"""Pytest root for UC26: puts uc26/ on sys.path (features, models, backtest, ingest are top-level packages)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
