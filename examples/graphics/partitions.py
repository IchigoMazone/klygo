"""Runnable example for graphics.partitions."""

from klygo import graphics

graphics.partitions([{"split": "train", "images": 300, "ratio_pct": 50.0}, {"split": "val", "images": 300, "ratio_pct": 50.0}], show=False)
