"""Find exact and perceptual duplicate images."""

from examples.mining._sample import sample_data
from klygo import mining

report = mining.find_duplicates(sample_data())
print(report.groups)
