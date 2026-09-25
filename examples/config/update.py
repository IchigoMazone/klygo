"""Create an updated configuration without mutating input."""
from klygo import config
original = {"model": {"batch": 16}}
updated = config.update(original, {"model": {"batch": 32}})
assert updated.model.batch == 32 and original["model"]["batch"] == 16
