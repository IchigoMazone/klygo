"""Filter images by target class names."""

from examples.mining._sample import sample_data
from klygo import mining

selected = mining.classes(sample_data(), ["person"], match="any", verbose=True)
print(len(selected))
