"""Runnable example for graphics.mining."""

from klygo import graphics

rows = [{"strategy": "S3", "positive_images": 93, "background_images": 157, "boxes": 210, "mean_conf": 0.80, "median_conf": 0.82}]
graphics.mining(rows, kind="counts", show=False)
