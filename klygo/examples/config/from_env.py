"""Overlay parsed environment variables."""
import os
from klygo import config
os.environ["APP_MODEL_BATCH"] = "32"
assert config.from_env(prefix="APP_", parse_values=True)["model"]["batch"] == 32
del os.environ["APP_MODEL_BATCH"]
