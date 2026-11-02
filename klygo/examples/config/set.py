"""Assign a nested value on an independent copy."""
from klygo import config
settings = config.set({}, "model.optimizer.lr", 0.001)
assert settings.model.optimizer.lr == 0.001
