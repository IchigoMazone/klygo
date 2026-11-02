"""Remove duplicate images without changing the original dataset."""

from examples.mining._sample import sample_data
from klygo import mining

data = sample_data()
unique = mining.deduplicate(data, keep="best")
print(len(data), len(unique))
