"""Validate required paths and value types."""
from klygo import config
settings = {"model": {"batch": 16}}
assert config.validate(settings, {"model.batch": int})
