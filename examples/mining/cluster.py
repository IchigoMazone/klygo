"""Cluster images with classical features."""

from examples.mining._sample import sample_data
from klygo import mining

result = mining.cluster(sample_data(), clusters=2, verbose=True)
print(result.groups)
