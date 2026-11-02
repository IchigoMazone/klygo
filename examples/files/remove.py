"""Executable example for klygo.files.remove."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    target = files.mkdir(files.path(directory) / "temporary")
    files.remove(target)
    print(files.exists(target))


# Additional cases
# missing_ok controls whether an absent path is an error.
with TemporaryDirectory() as directory:
    missing = files.path(directory) / "missing"
    files.remove(missing)
    try:
        files.remove(missing, missing_ok=False)
    except FileNotFoundError:
        print("missing path reported")
