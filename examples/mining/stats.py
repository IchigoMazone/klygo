"""Summarize a detection dataset."""

from examples.mining._sample import sample_data
from klygo import mining, utils

report = mining.stats(sample_data(), verbose=True)
utils.tabulate(report.to_dict())
