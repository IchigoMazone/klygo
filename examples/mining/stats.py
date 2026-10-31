"""Summarize a detection dataset."""

from examples.mining._sample import sample_data
from klygo import mining

report = mining.stats(sample_data())
report.show()
