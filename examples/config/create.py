"""Create a configuration file from defaults."""

from tempfile import TemporaryDirectory

from klygo import config, files

with TemporaryDirectory() as directory:
    created = config.create(files.path(directory) / "settings.yaml", verbose=False)
    assert created.model.name == "yolov8n"
