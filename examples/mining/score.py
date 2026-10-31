"""Rank images using annotation and image-quality features."""

from examples.mining._sample import sample_data
from klygo import mining

report = mining.score(sample_data())
print(report.values)
