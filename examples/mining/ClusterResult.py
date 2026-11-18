"""Inspect the ClusterResult result type."""

from examples.mining._sample import sample_data
from klygo import mining, utils

result = mining.cluster(sample_data(), clusters=2)
utils.tabulate(result.to_dict())
