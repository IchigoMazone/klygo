"""Runnable example for graphics.plot_partition_distribution."""

from klygo import graphics

graphics.plot_partition_distribution([{"split": "train", "images": 300, "ratio_pct": 50.0}, {"split": "val", "images": 300, "ratio_pct": 50.0}], show=False)
