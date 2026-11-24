"""Filter images by bounding-box count and relative area constraints."""

from examples.mining._sample import sample_data
from klygo import mining

selected = mining.boxes(sample_data(), min_boxes=1, verbose=True)
print(len(selected))
