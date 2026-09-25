"""Load a dot-accessible configuration."""
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config
with TemporaryDirectory() as directory:
    path = Path(directory) / "settings.json"
    config.save(path, {"model": {"batch": 16}}, verbose=False)
    assert config.load(path, verbose=False).model.batch == 16
