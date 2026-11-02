"""Export an in-memory configuration."""

from tempfile import TemporaryDirectory

from klygo import config, files

with TemporaryDirectory() as directory:
    output = config.export({"seed": 7}, files.path(directory) / "settings.json", verbose=False)
    assert files.is_file(output)
