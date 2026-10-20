"""Build independent defaults with deep overrides."""
from klygo import config
settings = config.defaults({"model": {"batch": 32}})
assert settings["model"]["batch"] == 32
