"""Use the stateful Config interface."""
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import Config
with TemporaryDirectory() as directory:
    manager = Config.create_default(Path(directory) / "settings.yaml", verbose=False)
    manager.read(verbose=False); manager.set("model.batch", 32)
    assert manager.get("model.batch") == 32
