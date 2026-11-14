"""Compare a source dataset with a selected subset."""

from examples.mining._sample import sample_data
from klygo import mining, utils

data = sample_data()
selected = mining.select(data, method="weighted", limit=2)
utils.tabulate(mining.compare(data, selected).to_dict())
