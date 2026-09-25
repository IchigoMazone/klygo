"""Create a configuration file from defaults."""
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config
with TemporaryDirectory() as directory:
    created = config.create(Path(directory) / "settings.yaml", verbose=False)
    assert created.model.name == "yolov8n"
