"""List flattened configuration keys."""
from klygo import config
assert config.keys({"model": {"batch": 16}}, flat=True) == ["model.batch"]
