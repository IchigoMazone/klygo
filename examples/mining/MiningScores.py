"""Inspect the MiningScores report type."""

from examples.mining._sample import sample_data
from klygo import mining

report = mining.score(sample_data(), weights={"rarity": 1.0})
report.show()
