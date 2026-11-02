"""Distinguish stored None from a missing key."""
from klygo import config
assert config.has({"model": {"name": None}}, "model.name")
assert not config.has({}, "model.name")
