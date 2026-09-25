"""List flattened configuration values."""
from klygo import config
assert config.values({"model": {"batch": 16}}, flat=True) == [16]
