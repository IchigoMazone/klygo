"""Compare a source dataset with a selected subset."""

from examples.mining._sample import sample_data
from klygo import mining

data = sample_data()
selected = mining.select(data, method="weighted", limit=2)
mining.compare(data, selected).show()
