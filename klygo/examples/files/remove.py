"""Executable example for klygo.files.remove."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    target = Path(directory) / "temporary"
    target.mkdir()
    files.remove(target)
    print(target.exists())


# Additional cases
# missing_ok controls whether an absent path is an error.
with TemporaryDirectory() as directory:
    missing = Path(directory) / "missing"
    files.remove(missing)
    try:
        files.remove(missing, missing_ok=False)
    except FileNotFoundError:
        print("missing path reported")

