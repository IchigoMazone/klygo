"""List flattened configuration items."""
from klygo import config
assert config.items({"model": {"batch": 16}}, flat=True) == [("model.batch", 16)]
