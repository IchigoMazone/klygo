"""Experiment reporting, multi-seed aggregation, and audit table builders.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/reports/Document_Reports_Overview.ipynb

Public API (7 functions across 2 submodules):
    Loaders (`loaders.py`):
        1. load(...)                Load training logs (results.csv, args.yaml) from a run.
        2. partition_summary(...)   Summarize image/box counts across partitions or splits.
        3. extract_hyperparams(...) Extract and cross-check args.yaml hyperparameters.

    Analysis (`analysis.py`):
        4. aggregate_runs(...)      Aggregate multi-seed runs into Mean ± Std tables and curves.
        5. threshold_audit(...)     Sweep confidence thresholds and verify optimal T*.
        6. mining_breakdown(...)    Summarize positive/background counts and confidence.
        7. rank_evaluations(...)    Rank evaluation summaries and compute baseline deltas.
"""

from .loaders import load, partition_summary, extract_hyperparams
from .analysis import aggregate_runs, threshold_audit, mining_breakdown, rank_evaluations

__all__ = [
    "load",
    "partition_summary",
    "extract_hyperparams",
    "aggregate_runs",
    "threshold_audit",
    "mining_breakdown",
    "rank_evaluations",
]
