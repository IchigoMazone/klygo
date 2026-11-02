"""Compare configuration leaf paths."""
from klygo import config
changes = config.diff({"batch": 16}, {"batch": 32, "seed": 7})
assert changes["added"] == {"seed": 7}
assert changes["modified"]["batch"] == {"from": 16, "to": 32}
