"""Reduce image-level class imbalance."""

from examples.mining._sample import sample_data
from klygo import mining

balanced = mining.balance(sample_data(), target="median")
print(mining.stats(balanced, quality=False).images_per_class)
