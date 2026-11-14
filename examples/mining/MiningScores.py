"""Inspect the MiningScores report type."""

from examples.mining._sample import sample_data
from klygo import mining, utils

report = mining.score(sample_data(), weights={"rarity": 1.0})
utils.tabulate(report.to_dict())
