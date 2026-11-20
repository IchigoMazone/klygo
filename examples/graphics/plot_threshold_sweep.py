"""Runnable example for graphics.plot_threshold_sweep."""

from klygo import graphics

audit = {"best_threshold": 0.55, "rows": [{"threshold": 0.3, "tp": 94, "fp": 21, "precision": 0.81, "recall": 0.76, "f1": 0.78}, {"threshold": 0.55, "tp": 88, "fp": 5, "precision": 0.94, "recall": 0.71, "f1": 0.81}]}
graphics.plot_threshold_sweep(audit, kind="curves", show=False)
