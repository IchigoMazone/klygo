"""Runnable example for graphics.plot_mining_summary."""

from klygo import graphics

rows = [{"strategy": "S3", "positive_images": 93, "background_images": 157, "boxes": 210, "mean_conf": 0.80, "median_conf": 0.82}]
graphics.plot_mining_summary(rows, kind="counts", show=False)
