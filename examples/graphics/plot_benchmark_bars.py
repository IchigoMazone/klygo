"""Runnable example for graphics.plot_benchmark_bars."""

from klygo import graphics

rows = [{"strategy": "S1", "mAP50-95_mean": 0.65, "mAP50-95_std": 0.01}, {"strategy": "S4", "mAP50-95_mean": 0.68, "mAP50-95_std": 0.006}]
graphics.plot_benchmark_bars(rows, show=False)
