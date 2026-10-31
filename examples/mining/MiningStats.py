"""Inspect the MiningStats report type."""

from examples.mining._sample import sample_data
from klygo import mining

report = mining.stats(sample_data(), quality=False)
print(report.to_dict())
