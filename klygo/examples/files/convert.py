"""Executable example for klygo.files.convert."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "data.json"
    target = Path(directory) / "data.yaml"
    files.save(source, {"value": 42}, verbose=False)
    files.convert(source, target, verbose=False)
    print(files.load(target, verbose=False))


# Additional cases
# Conversion keeps the decoded Python value unchanged.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source, target = root / "settings.yaml", root / "settings.json"
    files.save(source, {"enabled": True}, verbose=False)
    files.convert(source, target, verbose=False)
    assert files.load(target, verbose=False) == {"enabled": True}

