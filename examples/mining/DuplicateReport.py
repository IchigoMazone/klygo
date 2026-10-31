"""Inspect the DuplicateReport result type."""

from examples.mining._sample import sample_data
from klygo import mining

report = mining.find_duplicates(sample_data())
print(report.duplicate_images)
