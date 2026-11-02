"""Read nested values with a fallback."""
from klygo import config
settings = {"model": {"batch": 16}}
assert config.get(settings, "model.batch") == 16
assert config.get(settings, "model.epochs", 10) == 10
