"""Expand flat configuration paths."""
from klygo import config
assert config.unflatten({"model.batch": 16}) == {"model": {"batch": 16}}
