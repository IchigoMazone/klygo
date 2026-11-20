"""Experiment reporting, multi-seed aggregation, and audit table builders.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/reports/Document_Reports_Overview.ipynb

Public API (5 functions across 2 submodules):
    Loaders (`loaders.py`):
        1. load(...)                Load training logs (results.csv, args.yaml) from a run.
        2. partition_summary(...)   Summarize image/box counts across partitions or splits.

    Analysis (`analysis.py`):
        3. aggregate_runs(...)      Aggregate multi-seed runs into Mean ± Std tables and curves.
        4. threshold_audit(...)     Sweep confidence thresholds and verify optimal T*.
        5. mining_breakdown(...)    Summarize positive/background counts and confidence.
"""

from .loaders import load, partition_summary
from .analysis import aggregate_runs, threshold_audit, mining_breakdown

__all__ = [
    "load",
    "partition_summary",
    "aggregate_runs",
    "threshold_audit",
    "mining_breakdown",
]
