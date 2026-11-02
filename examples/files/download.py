"""Executable example for klygo.files.download."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source.txt"
    files.save(source, "klygo", verbose=False)
    downloaded = files.download(source, root / "downloads", verbose=False)
    print(downloaded, files.load(downloaded, verbose=False))


# Additional cases
# Existing destinations are protected unless overwrite=True.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "weights.txt"
    files.save(source, "v1", verbose=False)
    destination = files.download(source, root / "cache", verbose=False)
    try:
        files.download(source, root / "cache", verbose=False)
    except FileExistsError:
        print("destination protected:", destination)
