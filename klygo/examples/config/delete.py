"""Delete a nested key in place."""
from klygo import config
settings = {"model": {"batch": 16}}
assert config.delete(settings, "model.batch") and settings == {"model": {}}
