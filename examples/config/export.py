"""Export an in-memory configuration."""
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config
with TemporaryDirectory() as directory:
    assert config.export({"seed": 7}, Path(directory) / "settings.json", verbose=False).is_file()
