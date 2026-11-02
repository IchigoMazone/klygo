"""Inspect the MiningComparison report type."""

from examples.mining._sample import sample_data
from klygo import mining

data = sample_data()
report = mining.compare(data, mining.select(data, method="random", limit=2))
print(report.changes)
