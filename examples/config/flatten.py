"""Flatten a nested configuration."""
from klygo import config
assert config.flatten({"model": {"batch": 16}}) == {"model.batch": 16}
