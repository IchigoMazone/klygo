"""Select whole images for a balanced training subset."""

from examples.mining._sample import sample_data
from klygo import mining

selected = mining.select(sample_data(), method="weighted", limit=2, verbose=True)
print(len(selected))
