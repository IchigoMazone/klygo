"""Merge configurations from left to right."""
from klygo import config
merged = config.merge({"model": {"batch": 16}}, {"model": {"epochs": 10}})
assert merged.model.to_dict() == {"batch": 16, "epochs": 10}
