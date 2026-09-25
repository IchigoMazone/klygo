"""Save configuration data by destination extension."""
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config
with TemporaryDirectory() as directory:
    path = Path(directory) / "settings.yaml"
    assert config.save(path, {"debug": False}, verbose=False) == path
