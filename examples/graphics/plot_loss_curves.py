"""Runnable example for graphics.plot_loss_curves."""

from klygo import graphics

grouped = {"groups": {"S1": {"runs": [], "history_mean": {"epoch": [1, 2], "train/box_loss": [1.5, 1.1]}, "history_std": {"train/box_loss": [0.1, 0.05]}}}}
graphics.plot_loss_curves(grouped, metric="box_loss", show=False)
